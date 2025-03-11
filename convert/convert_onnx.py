import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_onnx_to_trt(onnx_path, trt_path):
    """Chuyển đổi mô hình ONNX sang TensorRT với hỗ trợ dynamic shape"""
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        profile = builder.create_optimization_profile()

        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("❌ Lỗi parse ONNX!")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return

        input_tensor = network.get_input(0)
        name = input_tensor.name
        shape = input_tensor.shape

        if -1 in shape:
            print(f" Phát hiện dynamic shape: {shape}")
            
            # Thiết lập min/opt/max shape
            min_shape = (1, 1, 224, 224)  # Nhỏ nhất (batch=1)
            opt_shape = (4, 1, 224, 224)  # Tối ưu (batch=4)
            max_shape = (8, 1, 224, 224)  # Lớn nhất (batch=8)

            profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Lỗi khi build TensorRT engine!")
            return

        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

        print(f" Convert thành công: {onnx_path} → {trt_path}")

if __name__ == "__main__":
    onnx_path = "./models/onnx/super_resolution.onnx"
    trt_path = "./models/super_resolution/1/model.plan"
    convert_onnx_to_trt(onnx_path, trt_path)
