#pragma once

#include <algorithm>
#include <istream>
#include <ostream>
#include <vector>
#include<math.h>
#include <thread>
#include <cstring> 
#include <mutex>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdint>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <string>
#include <vector>
#include <stddef.h> 
#include <ctime>
#include <iostream>
#include <dirent.h>
#include <random>
using namespace std;
typedef int prob_t;
typedef int mean_t;
typedef mean_t std_t;
typedef struct {
  prob_t prob1, prob2, prob3;
  mean_t mean1, mean2, mean3;
  std_t std1, std2, std3;
} gmm_t;


// ****** 码流IO
class BitInputStream final {
private:
    std::ifstream input;  
    int currentByte;
    int numBitsRemaining;
public:
    int read();
    int readNoEof();
    explicit BitInputStream(const char *binfile);
};


class BitArrayInputStream final {
	private:
    const uint8_t* data;    // 指向数据数组
    size_t dataSize;        // 数据长度
    size_t currentIndex;    // 当前读取的位置
    int currentByte;        // 当前的字节值
    int numBitsRemaining;   // 当前字节中剩余的位数

public:
    explicit BitArrayInputStream(const uint8_t* inputData, size_t size);
    int read();
    int readNoEof();
};


class BitOutputStream final {
	private: 
		int currentByte;
		int numBitsFilled;
	public:
		uint8_t* data_addr;
		size_t size=0;
		BitOutputStream(uint8_t* _addr);	
		BitOutputStream(size_t len);	
	    void write(int b);
		void finish();
};


// 函数 遍历txt文件(数据为每行一个)，返回对应的数组，数组长度，数据最大值，最小值
void read_txt(char file_path[],int16_t* &data, size_t &size, int &max, int &min);

class EncTable{
public:
	char exp_file_path[255] = "./table/exp.bin";
	char cdf_file_path[255] = "./table/cdf.bin";
	// 表数据
    uint16_t* exp_table = nullptr;
    size_t exp_size = 0;
    uint32_t* cdf_table = nullptr;
    size_t cdf_size = 0;	\
	// 表信息
	int exp_scale=1000,exp_x_bound=-12;
	int cdf_scale=10000,cdf_x_bound=-5;
	int scale_pred =10000;

	// 数据边界,左闭右闭
    int low_bound=0,high_bound=65536;
	long long freqs_resolution=10000000;

	// 要返回的结果
	uint32_t sym_low,sym_high,total_freqs;

	// f(low_bound-0.5)和f(high_bound+0.5)的值
	uint64_t l_bound = 0, r_bound = 0;

	// GMM参数
	int* probs=nullptr;
	int* means=nullptr;
	int* stds=nullptr;
	uint32_t prob_sum=0;

	EncTable(uint32_t freqs_resolution,int _low_bound,int _high_bound);
	void update(int m_probs[3],int m_means[3],int m_stds[3]);
	void get_bound(int x);
	// 析构函数
	~EncTable();
};


class DecTable{
public:
	// 文件路径
	char exp_file_path[255] = "./table/exp.bin";
	char cdf_file_path[255] = "./table/cdf.bin";
	// 表数据
    uint16_t* exp_table = nullptr;
    size_t exp_size = 0;
    uint32_t* cdf_table = nullptr;
    size_t cdf_size = 0;	\
	// 表信息
	int exp_scale=1000,exp_x_bound=-12;
	int cdf_scale=10000,cdf_x_bound=-5;
	int scale_pred = 10000;
	// 数据边界,左闭右闭
    int low_bound=0,high_bound=65536,freqs_resolution=1000000;

	// 要返回的结果
	int* sym_low=nullptr,total_freqs=0;
	// GMM参数
	int* probs=nullptr;
	int* means=nullptr;
	int* stds=nullptr;
	int prob_sum=0;

	DecTable(int freqs_resolution,int _low_bound,int _high_bound);
	void update(int m_probs[3],int m_means[3],int m_stds[3]);
	void get_bounds();
	// 析构函数
	~DecTable();
};


// ****** 编码器
class ArithmeticCoderBase {
	protected: int numStateBits;
		uint64_t fullRange;
		uint64_t halfRange;
		uint64_t quarterRange;
		uint64_t minimumRange;
		uint64_t maximumTotal;
		uint64_t stateMask;
		uint64_t low;
		uint64_t high;
	public: 
		explicit ArithmeticCoderBase(int numBits);
		virtual ~ArithmeticCoderBase() = 0;
		void update(uint32_t total, uint32_t symlow, uint32_t symhigh);
		virtual void shift() = 0;
		virtual void underflow() = 0;
};


class ArithmeticDecoder : private ArithmeticCoderBase {
	private: 
		BitInputStream input;
		uint64_t code;
	public: 
		ArithmeticDecoder(int numBits, const char *binfile);
		int read(DecTable &freqs);
		void shift() override;
		void underflow() override;
		int readCodeBit();
};


class ArithmeticEncoder final : private ArithmeticCoderBase {
	public: 
		uint8_t* bin;
		BitOutputStream output;
		unsigned long numUnderflow;
		explicit ArithmeticEncoder(int numBits, size_t len);
		void write(uint32_t total, uint32_t symLow, uint32_t symHigh);
		void finish();
		void shift() override;
		void underflow() override;
};

class ArithmeticBlockEncoder{
	private:
		ArithmeticEncoder enc;
		EncTable freqs_table;
		size_t len = 0, p=0;
	public:
		ArithmeticBlockEncoder(int numBits, size_t len, int16_t gmm_scale, int16_t xmin, int16_t xmax);
		vector<uint8_t> coding(vector<vector<int>> gmms, int16_t* data);
};

class ArithmeticPixelDecoder{
	private: 
		ArithmeticDecoder dec;
		DecTable freqs_table;
		size_t len = 0, p=0;
	public: 
		ArithmeticPixelDecoder(int numBits, const char *binfile, size_t len, int16_t gmm_scale, int16_t xmin, int16_t xmax);
		int read(vector<int> gmm);
};

vector<uint8_t> coding(vector<vector<vector<int>>>gmms, vector<vector<int16_t>>datas, vector<int16_t> gmm_scales);