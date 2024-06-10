#include <stddef.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

// clang-format will break include orders
// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/device_memory.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "broadcast_load_epilogue_c2x.hpp"
#include "common.hpp"
// clang-format on

using namespace cute;

/*
   This defines a quantized GEMM operation with dequantized output, similar to
   torch._scaled_mm. It is defined using the CUTLASS 2.x API, and is used for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).

   A and B may be both either int8 or fp8_e4m3. A can be quantized per-tensor or
   per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/

namespace {

template <typename ElementD, typename OutputTileThreadMap>
struct TrivialEpilogue {
 private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

 public:
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Accum>;
  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    return {};
  }
};

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogue {
 private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;

  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    auto tuple = std::make_tuple(args...);

    torch::Tensor const& a_scales = std::get<0>(tuple);
    torch::Tensor const& b_scales = std::get<1>(tuple);

    auto a_scales_ptr = a_scales.data_ptr<float>();
    auto b_scales_ptr = b_scales.data_ptr<float>();

    using ScaleAArgs = typename ScaleA::Arguments;
    using ScaleBArgs = typename ScaleB::Arguments;

    ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
    ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};

    typename EVTCompute0::Arguments evt0_compute_args{b_args};

    typename EVTCompute::Arguments evt_compute_args{a_args, evt0_compute_args};
    return evt_compute_args;
  }
};

template <typename Arch, typename ElementAB_, typename ElementD_,
          template <typename, typename> typename Epilogue_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages>
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                cutlass::arch::OpMultiplyAdd>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType = 
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, 16, 
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, 16, 
      float, cutlass::layout::RowMajor, 4,
      ElementAcc, float, cutlass::arch::OpClassTensorOp, 
      Arch, 
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                         torch::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,  // universal mode
      problem_size,                                           // problem size
      1,                                                      // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  // Launch the CUTLASS GEMM kernel.
  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;

  if (out.dtype() == torch::kInt8) {
    return cutlass_gemm_caller<
        cutlass_2x_gemm<cutlass::arch::Sm75, int8_t, int8_t, ScaledEpilogue,
                        TileShape, WarpShape, InstructionShape, 2>>(
        out, a, b, a_scales, b_scales);
  } else if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::bfloat16_t, ScaledEpilogue,
        TileShape, WarpShape, InstructionShape, 2>>(out, a, b, a_scales,
                                                    b_scales);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::half_t, ScaledEpilogue, TileShape,
        WarpShape, InstructionShape, 2>>(out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (out.dtype() == torch::kInt8) {
    return cutlass_gemm_caller<
        cutlass_2x_gemm<cutlass::arch::Sm80, int8_t, int8_t, ScaledEpilogue,
                        TileShape, WarpShape, InstructionShape, 5>>(
        out, a, b, a_scales, b_scales);
  } else if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::bfloat16_t, ScaledEpilogue,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                    b_scales);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::half_t, ScaledEpilogue, TileShape,
        WarpShape, InstructionShape, 5>>(out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales) {
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kInt8) {
      return cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, int8_t, int8_t, ScaledEpilogue,
                          TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, a_scales, b_scales);
    } else if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::bfloat16_t, ScaledEpilogue,
          TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                      b_scales);
    } else {
      assert(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::half_t, ScaledEpilogue,
          TileShape, WarpShape, InstructionShape, 5>>(out, a, b, a_scales,
                                                      b_scales);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kFloat8_e4m3fn) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
          ScaledEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, a_scales, b_scales);
    } else if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::bfloat16_t,
          ScaledEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::half_t,
          ScaledEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, a_scales, b_scales);
    }
  }
}

void cutlass_gemm_sm75(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;

  if (out.dtype() == torch::kInt8) {
    return cutlass_gemm_caller<
        cutlass_2x_gemm<cutlass::arch::Sm75, int8_t, int8_t, TrivialEpilogue,
                        TileShape, WarpShape, InstructionShape, 2>>(out, a, b);
  } else if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::bfloat16_t, TrivialEpilogue,
        TileShape, WarpShape, InstructionShape, 2>>(out, a, b);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::half_t, TrivialEpilogue,
        TileShape, WarpShape, InstructionShape, 2>>(out, a, b);
  }
}

void cutlass_gemm_sm80(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (out.dtype() == torch::kInt8) {
    return cutlass_gemm_caller<
        cutlass_2x_gemm<cutlass::arch::Sm80, int8_t, int8_t, TrivialEpilogue,
                        TileShape, WarpShape, InstructionShape, 5>>(out, a, b);
  } else if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::bfloat16_t, TrivialEpilogue,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::half_t, TrivialEpilogue,
        TileShape, WarpShape, InstructionShape, 5>>(out, a, b);
  }
}

void cutlass_gemm_sm89(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b) {
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kInt8) {
      return cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, int8_t, int8_t, TrivialEpilogue,
                          TileShape, WarpShape, InstructionShape, 5>>(out, a,
                                                                      b);
    } else if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::bfloat16_t, TrivialEpilogue,
          TileShape, WarpShape, InstructionShape, 5>>(out, a, b);
    } else {
      assert(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::half_t, TrivialEpilogue,
          TileShape, WarpShape, InstructionShape, 5>>(out, a, b);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kFloat8_e4m3fn) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
          TrivialEpilogue, TileShape, WarpShape, InstructionShape, 5>>(out, a,
                                                                       b);
    } else if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::bfloat16_t,
          TrivialEpilogue, TileShape, WarpShape, InstructionShape, 5>>(out, a,
                                                                       b);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::half_t,
          TrivialEpilogue, TileShape, WarpShape, InstructionShape, 5>>(out, a,
                                                                       b);
    }
  }
}
