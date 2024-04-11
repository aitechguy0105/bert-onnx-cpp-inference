#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <locale>
#include <codecvt>
#include <string>
#include <sstream>
#include "unilib/tokenizer.h"
BertTokenizer tokenizer;
BasicTokenizer basictokenizer(true);
int max_seq_length = 512;
std::string vocab_dir = "D:/work/AI/nlp/BERT/onnx/vocab.txt";
std::string onnxmodel_dir = "D:/work/AI/nlp/BERT/onnx/model.onnx";
vector <std::string> id2label = { "O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC" };
void tokenize(string text, vector<string>& tokens, vector<int64_t>& valid_positions)
{
    vector<string> words = basictokenizer.tokenize(text);
    vector<string> token;
    vector<string>::iterator itr;
    for (itr = words.begin(); itr < words.end(); itr++)
    {
        token = tokenizer.tokenize(*itr);
        tokens.insert(tokens.end(), token.begin(), token.end());
        for (int i = 0; i < token.size(); i++)
        {
            if (i == 0)
                valid_positions.push_back(1);
            else
                valid_positions.push_back(0);
        }
    }
}

void preprocess(string text, vector<int64_t>& input_ids, vector<int64_t>& input_mask, vector<int64_t>& segment_ids, vector<int64_t>& valid_positions)
{
    vector<string> tokens;
    tokenize(text, tokens, valid_positions);
    // insert "[CLS}"
    tokens.insert(tokens.begin(), "[CLS]");
    valid_positions.insert(valid_positions.begin(), 1.0);
    // insert "[SEP]"
    tokens.push_back("[SEP]");
    valid_positions.push_back(1.0);
    for (int i = 0; i < tokens.size(); i++)
    {
        segment_ids.push_back(0.0);
        input_mask.push_back(1.0);
    }
    input_ids = tokenizer.convert_tokens_to_ids(tokens);

    

    //while (input_ids.size() < max_seq_length)
    //{
    //    input_ids.push_back(0.0);
    //    input_mask.push_back(0.0);
    //    segment_ids.push_back(0.0);
    //    valid_positions.push_back(0.0);
    //}
}
vector<map<string, string>> predict(string& text, Ort::Session& session,Ort::MemoryInfo& memory_info, std::vector<const char*>*  input_node_names, std::vector<const char*>* output_node_names) {
    std::vector<Ort::Value> inputTensor;        // Onnxruntime allowed input
    std::vector<int64_t> valid_positions;

    std::vector<int64_t> input_data;
    std::vector<int64_t> attension_mask;
    std::vector<int64_t> token_type_ids;
    std::vector<int64_t> input_shape;

    //preprocess(text, input_ids, input_mask, segment_ids, valid_positions);
    preprocess(text, input_data, attension_mask, token_type_ids, valid_positions);
    input_shape = { 1, (int64_t)(input_data.size()) };
    //[0] 101	__int64
//    [1]	100	__int64
//    [2]	3632	__int64
//    [3]	2000	__int64
//    [4]	100	__int64
//    [5]	1012	__int64
//    [6]	102	__int64

    //input_data = { 101, 3889, 3632, 2000, 3000, 1012, 102 };

    try {

        inputTensor.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
        inputTensor.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, attension_mask.data(), attension_mask.size(), input_shape.data(), input_shape.size()));
        inputTensor.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, token_type_ids.data(), token_type_ids.size(), input_shape.data(), input_shape.size()));

    }
    catch (Ort::Exception oe) {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        
    }

    std::vector<Ort::Value> OutputTensor;
    try {

        OutputTensor = session.Run(Ort::RunOptions{ nullptr }, input_node_names->data(), inputTensor.data(), inputTensor.size(), output_node_names->data(), 1);

    }
    catch (Ort::Exception oe) {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";

    }
    //printf("%d", OutputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount());

    std::vector<int> argmaxIds;
    for (const auto& outputValue : OutputTensor) {
        // Get the pointer to the output data
        std::vector<int64_t> shapeInfo = OutputTensor.front().GetTensorTypeAndShapeInfo().GetShape();

        // Get the pointer to the output data
        const float* outputData = OutputTensor.front().GetTensorData<float>();

        int dim1 = shapeInfo[0];
        int dim2 = shapeInfo[1];
        int dim3 = shapeInfo[2];
        int maxVal = 0;
        int argMaxId = -1;

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                maxVal = 0;
                argMaxId = -1;
                for (int k = 0; k < dim3; k++) {
                    // Access each item of the output tensor
                    float item = outputData[i * dim2 * dim3 + j * dim3 + k];
                    // Do something with the item, such as printing it
                    // std::cout << "Output item [" << i << "][" << j << "][" << k << "]: " << item << std::endl;

                    float currentValue = outputData[j * dim3 + k];
                    if (currentValue > maxVal) {
                        maxVal = currentValue;
                        argMaxId = k;

                    }
                }
                argmaxIds.push_back(argMaxId);
            }
        }
    }


    for (auto item : argmaxIds) {
        //std::cout << id2label[item] << std::endl;
    }

    vector<string> words = basictokenizer.tokenize(text);
    vector<map<string, string>> result;
    for (int i = 0; i < words.size(); i++)
    {
        map<string, string> mp;
        mp.insert({ words.at(i),id2label[argmaxIds[i + 1]] });
        result.push_back(mp);
    }
    return result;
}

int main() {
    //check providers
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        //std::cout << provider << std::endl;
    }

    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cuda_options;

    sessionOptions.SetIntraOpNumThreads(1);
    // Optimization will take time and memory during startup
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    // CUDA options. If used.
    bool  _UseCuda = false;


    // Convert narrow string to wide string
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    const std::wstring wideString = converter.from_bytes(onnxmodel_dir);
    // Convert wide string to wchar_t*
    const wchar_t* ModelPath = wideString.c_str();
    if (_UseCuda)
    {
        cuda_options.device_id = 0;  //GPU_ID
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
        cuda_options.arena_extend_strategy = 0;
        // May cause data race in some condition
        cuda_options.do_copy_in_default_stream = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
    }
    Ort::Session session = Ort::Session(env, ModelPath, sessionOptions);
    /*try {
        
            }
    catch (Ort::Exception oe) {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }*/

    Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input
    try {
        memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception oe) {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    // Demonstration of getting input node info by code
    size_t num_input_nodes = 0;
    std::vector<const char*>* input_node_names = nullptr; // Input node names
    std::vector<std::vector<int64_t>> input_node_dims;    // Input node dimension.
    ONNXTensorElementDataType type;                       // Used to print input info
    Ort::TypeInfo* type_info;

    num_input_nodes = session.GetInputCount();
    input_node_names = new std::vector<const char*>;
    bool _Debug = true;
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++) {

        char* tempstring = new char[strlen(session.GetInputNameAllocated(i, allocator).get()) + 1];
        strcpy_s(tempstring, strlen(session.GetInputNameAllocated(i, allocator).get()) + 1, session.GetInputNameAllocated(i, allocator).get());
        input_node_names->push_back(tempstring);
        type_info = new Ort::TypeInfo(session.GetInputTypeInfo(i));
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        type = tensor_info.GetElementType();
        input_node_dims.push_back(tensor_info.GetShape());

        // print input shapes/dims
        
        //if (_Debug) {
        //    printf("Input %d : name=%s\n", i, input_node_names->back());
        //    printf("Input %d : num_dims=%zu\n", i, input_node_dims.back().size());
        //    for (int j = 0; j < input_node_dims.back().size(); j++)
        //        printf("Input %d : dim %d=%jd\n", i, j, input_node_dims.back()[j]);
        //    printf("Input %d : type=%d\n", i, type);
        //}
        delete(type_info);
    }

    // Set output node name explicitly
    std::vector<const char*>* output_node_names;
    std::vector<const char*> output_string = { "logits" };

    // Convert output_string to const char* and add it to output_node_names
    output_node_names = &output_string;




    // this will make the input into 1,3,640,640
    //cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), (0, 0, 0), false, false);
    
    tokenizer.add_vocab(vocab_dir.c_str());
    map<string, string> label2extract = {
        {"B-PER","Person"},
        { "B-ORG", "Organization" },
        { "B-LOC", "Location" },
        { "B-MISC", "Miscellaneous" }
    };
    vector<string> vecMask = { "PER", "ORG", "LOC","MISC" };
    string text;
    int mode; // 0: ai_extract, 1: ai_mask
    std::string input;
    std::stringstream ss;
    while (true)
    {
        cout << "Enter a number (0: ai_extract, 1: ai_mask): ";
        
        do {
            std::cin >> input;

            ss = std::stringstream(input);
            ss >> mode;
            if (ss.eof())
                break;
            // Input is not a valid integer
            std::cout << "You entered a string: " << input << std::endl << "Enter a valid number: ";

        } while (!ss.eof());
        
        cin.ignore();
        cout << "\n" << "Input -> ";
        getline(cin, text);
        std::vector<map<string, string>> result = predict(text, session, memory_info, input_node_names, output_node_names);
        for (auto it = result.begin(); it != result.end(); ++it) {
            auto& tokens = *it;
            for (const auto& pair : tokens) {

                if (mode == 1) {
                    if (label2extract.find(pair.second) != label2extract.end()) {
                        std::cout << "[MASK] ";
                    }
                    else {
                        std::cout << pair.first << " ";
                    }
                                            
                }
                else if (mode == 0) {
                    if (label2extract.find(pair.second) != label2extract.end())
                        std::cout << pair.first << " : " << label2extract[pair.second] << std::endl;
                }
                                
            }

        }
        cout << endl;
    }

    return 1;	
}