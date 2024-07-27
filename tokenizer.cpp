#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <memory>
#include <algorithm>
#include <future>


std::vector<uint16_t> loadBytesFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary); // Open in binary mode

    file.seekg(0, std::ios::end); 
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg); // Rewind back to the beginning

    std::vector<uint8_t> bytes(fileSize);  
    file.read(reinterpret_cast<char*>(bytes.data()), fileSize);
    std::vector<uint16_t> bytess(fileSize);

    std::transform(bytes.begin(), bytes.end(), bytess.begin(), 
                    [](uint8_t c) { return static_cast<uint16_t>(c); });
    return bytess;
}

class BPETokenizer {
private:
    int vocab_size;
    const int BASE_VOCAB_SIZE = 256;
    
    struct TokenPair {
        uint16_t first;
        uint16_t second;
        
        bool operator==(const TokenPair& other) const {
            return first == other.first && second == other.second;
        }
    };
    
    struct TokenPairHash {
        std::size_t operator()(const TokenPair& k) const {
            return (static_cast<size_t>(k.first) << 16) | k.second;
        }
    };
    
    struct Merge {
        TokenPair pair;
        uint16_t token = 0;
    };
    
    std::vector<Merge> merges;
    
    struct TrieNode {
        std::unordered_map<uint16_t, std::unique_ptr<TrieNode>> children;
        uint16_t token = 0;
    };
    std::unique_ptr<TrieNode> encode_trie;

    // std::unordered_map<uint16_t, std::pair<uint16_t, uint16_t>> reverse_merges;
public:
    BPETokenizer(int size) : vocab_size(size) {
        encode_trie = std::make_unique<TrieNode>();
    }

    void train(std::vector<uint16_t>& tokens, bool verbose = false) {
        std::cout << "Training..." << std::endl;
        int n_merges = vocab_size - BASE_VOCAB_SIZE - merges.size();
        std::unordered_map<TokenPair, int, TokenPairHash> occurances; 

        for (int i = 0; i < n_merges; i++) {
            int len = tokens.size();
            if (len == 1) {
                std::cout << "No more tokens to merge" << std::endl;
                break;
            }

            for (auto& entry : occurances) {
                entry.second = 0;
            }

            // Parallel Occurrence Counting
            #pragma omp parallel num_threads(std::thread::hardware_concurrency())
            {
                std::unordered_map<TokenPair, int, TokenPairHash> local_occurances;

                #pragma omp for
                for (int b = 0; b < len - 1; b++) {
                    TokenPair pair = {tokens[b], tokens[b + 1]};
                    local_occurances[pair]++;
                }

                #pragma omp critical
                {
                    for (const auto& entry : local_occurances) {
                        occurances[entry.first] += entry.second;
                    }
                }
            }

            // Find Best Pair
            TokenPair best_pair;
            int max_occur = 0;
            for (const auto& entry : occurances) {
                if (entry.second > max_occur) {
                    max_occur = entry.second;
                    best_pair = entry.first;
                }
            }

            uint16_t new_token = BASE_VOCAB_SIZE + i + 1;
            Merge new_merge = {best_pair, new_token};
            merges.push_back(new_merge); 

            // Merge
            int writeIndex = 0;
            for (int b = 0; b < len - 1; b++) {
                TokenPair pair = {tokens[b], tokens[b+1]};
                if (pair == best_pair) {
                    tokens[writeIndex++] = new_token;
                    b++;
                } else {
                    tokens[writeIndex++] = tokens[b];
                }
            }
            tokens.resize(writeIndex);

            if (verbose || i % 100 == 0) {
                std::cout << "Merge " << i + 1 << "/" << n_merges << ": " 
                        << best_pair.first << " + " << best_pair.second << " -> " << new_token << std::endl;
            }
        }
        std::cout << "Tokenizer training completed" << std::endl;
        buildEncodeTrie();
    }


    void buildEncodeTrie() {
        encode_trie = std::make_unique<TrieNode>();
        for (const auto& merge : merges) {
            TrieNode* node = encode_trie.get();
            
            if (!node->children[merge.pair.first]) {
                node->children[merge.pair.first] = std::make_unique<TrieNode>();
            }
            node = node->children[merge.pair.first].get();
            
            if (!node->children[merge.pair.second]) {
                node->children[merge.pair.second] = std::make_unique<TrieNode>();
            }
            node = node->children[merge.pair.second].get();
            
            node->token = merge.token;

            // reverse_merges[merge.token] = {merge.pair.first, merge.pair.second};
        }
    }

    std::vector<uint16_t> encode(const std::vector<uint16_t>& tokens, bool verbose = true) {
        std::vector<uint16_t> encoded;
        encoded.reserve(tokens.size());

        size_t i = 0;
        const size_t total_tokens = tokens.size();
        const size_t update_interval = total_tokens / 100;
        size_t last_update = 0;

        if (verbose) std::cout << "Encoding progress: 0%" << std::flush;

        size_t c = 0;
        while (i < total_tokens) {
            const TrieNode* node = encode_trie.get();
            size_t j = i;
            while (j < total_tokens && node->children.count(tokens[j])) {
                node = node->children.at(tokens[j]).get();
                j++;
            }
            if (node->token != 0) {
                encoded.push_back(node->token);
                i = j;
                c++;
            } else {
                encoded.push_back(tokens[i]);
                i++;
                c++;
            }

            if (i - last_update >= update_interval) {
                int progress = static_cast<int>((i * 100) / total_tokens);
                if (verbose) std::cout << "\rEncoding progress: " << progress << "%" << std::flush;
                last_update = i;
            }
        }
        size_t s = encoded.size();
        return encoded;
    }

    std::vector<uint16_t> encode(const std::vector<uint16_t>& tokens) {
        // wip
    }

    int saveMerges(const std::string& file_name) {
        std::ofstream vocab_file(file_name);
        if (!vocab_file) {
            std::cerr << "Error opening vocabulary file." << std::endl;
            return 0;
        }

        for (const auto& merge : merges) {
            vocab_file << merge.pair.first << " " 
                    << merge.pair.second << " " 
                    << merge.token << "\n";
        }

        vocab_file.close();
        if (vocab_file.fail()) {
            std::cerr << "Error writing to vocabulary file." << std::endl;
            return 0;
        }

        std::cout << "Vocabulary saved as " << file_name << std::endl;
        return 1;
    }

    int loadMerges(const std::string& file_name) {
        std::ifstream vocab_file(file_name);
        if (!vocab_file) {
            std::cerr << "Error opening merges file." << std::endl;
            return 0;
        }

        merges.clear();

        uint16_t first, second, token;
        while (vocab_file >> first >> second >> token) {
            TokenPair pair{first, second};
            merges.push_back({pair, token});
        }

        if (vocab_file.bad()) {
            std::cerr << "Error reading from merges file." << std::endl;
            return 0;
        }

        vocab_file.close();

        // Rebuild the encode trie
        buildEncodeTrie();

        std::cout << "Merges loaded from " << file_name << std::endl;
        return 1;
    }
};

std::string loadFile(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file) {
        std::cerr << "Error opening input file." << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string text = buffer.str();
    file.close();  

    return text;
}

int saveEncoded(const std::vector<uint16_t>& encoded, const std::string& file_name) {
    std::ofstream encoded_file(file_name, std::ios::binary);
    if (!encoded_file) {
        std::cerr << "Error opening encoded output file." << std::endl;
        return 0;
    }

    for (const auto& token : encoded) {
        encoded_file.write(reinterpret_cast<const char*>(&token), sizeof(token));
    }

    encoded_file.close();
    std::cout << "Encoded file saved as " << file_name << std::endl;
    return 1;
}

std::vector<uint16_t> encodeChunk(BPETokenizer& tokenizer, const std::vector<uint16_t>& input, size_t start, size_t end, bool verbose = true) {
    auto chunk = std::vector<uint16_t>(input.begin() + start, input.begin() + end);
    return tokenizer.encode(chunk, verbose);
}

std::vector<uint16_t> parallelEncode(BPETokenizer& tokenizer, const std::vector<uint16_t>& input, const size_t desired_num_threads) {
    const size_t avail_num_threads = std::thread::hardware_concurrency();
    const size_t num_threads = (desired_num_threads == -1 || avail_num_threads < desired_num_threads) ? avail_num_threads : desired_num_threads;
    const size_t chunk_size = input.size() / num_threads;

    std::vector<std::future<std::vector<uint16_t>>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? input.size() : (i + 1) * chunk_size;
        futures.emplace_back(std::async(std::launch::async, encodeChunk, std::ref(tokenizer), std::ref(input), start, end, num_threads == 1));
    }

    std::vector<uint16_t> output;
    for (auto& future : futures) {
        auto chunk = future.get();
        output.insert(output.end(), chunk.begin(), chunk.end());
    }

    std::cout << "\nEncoding completed. Original size: " << input.size()
              << ", Encoded size: " << output.size()
              << ", Compression ratio: " << (static_cast<float>(output.size()) / input.size() ) << std::endl;

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <mode> <input_file> [options]" << std::endl;
        std::cout << "Modes:" << std::endl;
        std::cout << "  train <vocab_size>" << std::endl;
        std::cout << "  encode" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -m, --merges <merges_file>  Specify merges file (required for encode mode)" << std::endl;
        std::cout << "  -o, --output <output_file>  Specify output file (default: output.bin for encode, merges.txt for train)" << std::endl;
        std::cout << "  -t, --threads <n>  Specify number of threads to use (-1 for all available) (default: 1)" << std::endl;
        
        return 1;
    }

    std::string mode = argv[1];
    std::string input_file = argv[2];
    std::string merges_file;
    std::string output_file;
    int vocab_size = 0;
    int threads = 1;
    bool silent = false;
    if (mode == "train") {
        if (argc < 4) {
            std::cout << "Error: Vocabulary size required for train mode" << std::endl;
            return 1;
        }
        vocab_size = std::atoi(argv[3]);
        if (vocab_size <= 256) {
            std::cout << "Error: Vocabulary size must be greater than 256" << std::endl;
            return 1;
        }
    } else if (mode != "encode") {
        std::cout << "Error: Invalid mode. Use 'train' or 'encode'" << std::endl;
        return 1;
    }

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--merges") {
            if (i + 1 < argc) {
                merges_file = argv[++i];
            } else {
                std::cout << "Error: Merges file name missing" << std::endl;
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cout << "Error: Output file name missing" << std::endl;
                return 1;
            }
        } else if (arg == "-w" || arg == "--workers") {
            if (i + 1 < argc) {
                threads = std::atoi(argv[++i]);
            } else {
                std::cout << "Error: Number of workers missing" << std::endl;
                return 1;
            }
            if (threads != -1 && threads <= 0) {
                std::cout << "Error: Number of workers should be -1 or positive" << std::endl;
                return 1;                
            }
        }
    }

    if (mode == "encode" && merges_file.empty()) {
        std::cout << "Error: Merges file required for encode mode" << std::endl;
        return 1;
    }

    if (output_file.empty()) {
        output_file = (mode == "encode") ? "output.bin" : "merges.txt";
    }

    std::vector<uint16_t> input_bytes = loadBytesFromFile(input_file);
    if (input_bytes.empty()) {
        return 1;
    }

    BPETokenizer tokenizer(vocab_size);

    if (mode == "train") {
        tokenizer.train(input_bytes);
        if (!tokenizer.saveMerges(output_file)) {
            return 1;
        }
    } else {  // encode mode
        if (!tokenizer.loadMerges(merges_file)) {
            return 1;
        }
        // input_bytes = threads != 1 ? parallelEncode(tokenizer, input_bytes, threads) : tokenizer.encode(input_bytes);
        input_bytes = parallelEncode(tokenizer, input_bytes, threads);
        
        if (!saveEncoded(input_bytes, output_file)) {
            return 1;
        }
    }

    return 0;
}