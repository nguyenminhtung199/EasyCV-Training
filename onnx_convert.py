import re
import torch
import argparse

from onnx_helper import ONNXProcessedModelMultiTask, ONNXProcessedModel
from onnx import helper, TensorProto, shape_inference
import onnx

def fix_flatten_dynamic_shapes(onnx_path: str, fixed_onnx_path: str):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)

    for node in list(model.graph.node):
        if node.op_type == "Flatten":
            input_name = node.input[0]
            output_name = node.output[0]

            # Tạo tensor shape explicit [0, -1] để flatten toàn bộ batch
            reshape_shape_name = output_name + "_shape"
            reshape_shape_tensor = helper.make_tensor(
                name=reshape_shape_name,
                data_type=TensorProto.INT64,
                dims=[2],
                vals=[0, -1],
            )
            model.graph.initializer.append(reshape_shape_tensor)

            # Tạo node Reshape mới
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[input_name, reshape_shape_name],
                outputs=[output_name],
                name=node.name + "_fixed"
            )

            # Thay Flatten bằng Reshape explicit
            model.graph.node.remove(node)
            model.graph.node.append(reshape_node)

    onnx.save(model, fixed_onnx_path)
    print(f"✅ Saved fixed ONNX model to {fixed_onnx_path}")

def convert_onnx(file_path, multitask, fix_dynamic_shapes=False):
    print("Convert ONNX file: ", file_path)
    model = torch.load(file_path)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if multitask:
        model = ONNXProcessedModelMultiTask(model)
    else: 
        model = ONNXProcessedModel(model)
    model.to(device)
    model.eval()
    output_file = file_path.replace(".pt",".onnx")
    batch_size = 1
    # Input to the model
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=False).to(device)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               
                    x,                         
                    output_file,  
                    export_params=True,        
                    opset_version=13,          
                    do_constant_folding=True, 
                    input_names = ['input'],   
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'},   
                                    'output' : {0 : 'batch_size'}})
    if fix_dynamic_shapes:
        fix_flatten_dynamic_shapes(output_file, output_file)
    print(f"Save successful model onnx on {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, required=True, help="pytorch model path")
    parser.add_argument("-m", "--multitask", type=str, default=False, help="multitask model")
    parser.add_argument("-d", "--fix_dynamic_shapes", action='store_true', default=False, help="Fix dynamic shapes in Flatten nodes")
    args = parser.parse_args()
    convert_onnx(args.file_path, args.multitask, args.fix_dynamic_shapes)