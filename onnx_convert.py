import torch
import argparse

from onnx_helper import ONNXProcessedModelMultiTask, ONNXProcessedModel

def main(args):
    print(args.file_path)
    model = torch.load(args.file_path)
    model.eval()
    if args.multitask:
        model = ONNXProcessedModelMultiTask(model)
    else: 
        model = ONNXProcessedModel(model)
    model.eval()
    output_file = args.file_path.replace(".pt",".onnx")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    print(f"Save successful model onnx on {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, help="pytorch model path")
    parser.add_argument("-m", "--multitask", type=str, default=False, help="pytorch model path")
    main(parser.parse_args())