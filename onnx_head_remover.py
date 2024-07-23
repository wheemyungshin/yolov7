import onnx, os
import onnx_graphsurgeon as gs 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='mobilenet_128', help='model type')
    parser.add_argument('--dir', type=str, default='weights', help='initial weights path')
    parser.add_argument('--model-name', type=str, default='GF_phone_mobilenet_e099_no_opt_128_128.onnx', help='model name')
    parser.add_argument('--save-tag', type=str, default='modified_', help='save file tag')


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

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))

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

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))

    elif opt.type == 'yolov10_no_psa':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_487 = onnx.helper.make_tensor_value_info('487', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_507 = onnx.helper.make_tensor_value_info('507', onnx.TensorProto.FLOAT, [1, 18, 8, 8])
        output_527 = onnx.helper.make_tensor_value_info('527', onnx.TensorProto.FLOAT, [1, 18, 4, 4])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_377':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_376':
                graph.node.remove(nodes[i])        
            elif nodes[i].name == 'Conv_362':        
                graph.output.insert(i, output_527)

            elif nodes[i].name == 'Transpose_361':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_360':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_346':
                graph.output.insert(i, output_507)

            elif nodes[i].name == 'Transpose_345':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_344':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_330':
                graph.output.insert(i, output_487)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))
        
    elif opt.type == 'yolov7-tiny-liter_nomp':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_161 = onnx.helper.make_tensor_value_info('161', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_181 = onnx.helper.make_tensor_value_info('181', onnx.TensorProto.FLOAT, [1, 18, 8, 8])
        output_201 = onnx.helper.make_tensor_value_info('201', onnx.TensorProto.FLOAT, [1, 18, 4, 4])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_109':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_108':
                graph.node.remove(nodes[i])        
            elif nodes[i].name == 'Conv_94':        
                graph.output.insert(i, output_201)

            elif nodes[i].name == 'Transpose_93':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_92':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_78':
                graph.output.insert(i, output_181)

            elif nodes[i].name == 'Transpose_77':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_76':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_62':
                graph.output.insert(i, output_161)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))

    elif opt.type == 'tiny-lite_half_nomp':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_225 = onnx.helper.make_tensor_value_info('225', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_245 = onnx.helper.make_tensor_value_info('245', onnx.TensorProto.FLOAT, [1, 18, 8, 8])
        output_265 = onnx.helper.make_tensor_value_info('265', onnx.TensorProto.FLOAT, [1, 18, 4, 4])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_145':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_144':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_130':
                graph.output.insert(i, output_265)

            elif nodes[i].name == 'Transpose_129':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_128':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_114':
                graph.output.insert(i, output_245)

            elif nodes[i].name == 'Transpose_113':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_112':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_98':
                graph.output.insert(i, output_225)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))

    elif opt.type == 'mobilenetv4':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_580 = onnx.helper.make_tensor_value_info('580', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_600 = onnx.helper.make_tensor_value_info('600', onnx.TensorProto.FLOAT, [1, 18, 8, 8])
        output_620 = onnx.helper.make_tensor_value_info('620', onnx.TensorProto.FLOAT, [1, 18, 4, 4])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_211':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_210':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_196':
                graph.output.insert(i, output_620)

            elif nodes[i].name == 'Transpose_195':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_194':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_180':
                graph.output.insert(i, output_600)

            elif nodes[i].name == 'Transpose_179':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_178':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_164':
                graph.output.insert(i, output_580)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))
        
    elif opt.type == 'tiny_n78':
        graph.output.remove(graph.output[2])
        graph.output.remove(graph.output[1])
        graph.output.remove(graph.output[0])

        print(dir(onnx.TensorProto))

        output_282 = onnx.helper.make_tensor_value_info('282', onnx.TensorProto.FLOAT, [1, 18, 16, 16])
        output_302 = onnx.helper.make_tensor_value_info('302', onnx.TensorProto.FLOAT, [1, 18, 8, 8])
        output_322 = onnx.helper.make_tensor_value_info('322', onnx.TensorProto.FLOAT, [1, 18, 4, 4])

        for i in range(len(nodes)-1, 0, -1) :
            if nodes[i].name == 'Transpose_174':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_173':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_159':
                graph.output.insert(i, output_322)

            elif nodes[i].name == 'Transpose_158':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_157':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_143':
                graph.output.insert(i, output_302)

            elif nodes[i].name == 'Transpose_142':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Reshape_141':
                graph.node.remove(nodes[i])
            elif nodes[i].name == 'Conv_127':
                graph.output.insert(i, output_282)

        for node in nodes:
            print(node.name, ' : ', node.output)

        onnx.save(onnx_model, os.path.join(path, opt.save_tag+model_name))