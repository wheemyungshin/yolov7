import onnx, os
import onnx_graphsurgeon as gs 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='mobilenet_128', help='model type')
    parser.add_argument('--dir', type=str, default='weights', help='initial weights path')
    parser.add_argument('--model-name', type=str, default='GF_phone_mobilenet_e099_no_opt_128_128.onnx', help='model name')


    opt = parser.parse_args()

    path = opt.dir
    model_name = opt.model_name
    onnx_model = onnx.load(os.path.join(path,model_name))
    graph = onnx_model.graph
    # Get the nodes of the graph
    nodes = graph.node
    print(len(nodes))
    for node in nodes:
        print(node.name, ' : ', node.output)

    for j, output in enumerate(graph.output):
        print(j, " : ", output)

    if opt.type == 'mobilenet_128':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_666 = onnx.helper.make_tensor_value_info('666', onnx.TensorProto.FLOAT, [1, 18, 32, 32])
        output_686 = onnx.helper.make_tensor_value_info('686', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_706 = onnx.helper.make_tensor_value_info('706', onnx.TensorProto.FLOAT, [1, 18, 8, 8])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_242':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_241':
                graph.node.remove(nodes[i])        
            elif nodes[i].name == 'Conv_227':        
                graph.output.insert(i, output_706)

            elif nodes[i].name == 'Transpose_226':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_225':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_211':
                graph.output.insert(i, output_686)

            elif nodes[i].name == 'Transpose_210':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_209':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_195':
                graph.output.insert(i, output_666)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, 'modified_'+model_name))

    elif opt.type == 'mobilenet_v2_192':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_666 = onnx.helper.make_tensor_value_info('666', onnx.TensorProto.FLOAT, [1, 18, 24, 24])
        output_686 = onnx.helper.make_tensor_value_info('686', onnx.TensorProto.FLOAT, [1, 18, 12, 12])
        output_706 = onnx.helper.make_tensor_value_info('706', onnx.TensorProto.FLOAT, [1, 18, 6, 6])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_242':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_241':
                graph.node.remove(nodes[i])        
            elif nodes[i].name == 'Conv_227':        
                graph.output.insert(i, output_706)

            elif nodes[i].name == 'Transpose_226':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_225':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_211':
                graph.output.insert(i, output_686)

            elif nodes[i].name == 'Transpose_210':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_209':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_195':
                graph.output.insert(i, output_666)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, 'modified_'+model_name))