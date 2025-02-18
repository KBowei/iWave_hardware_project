#include <algorithm> 
// #include <cstdint> 
#include <cstdlib>   
#include <fstream>   
#include <iostream>  
#include <limits>   
#include <stdexcept> 
#include <string>   
#include <vector>    
#include "arithmetic_coding.hpp"


void read_txt(char file_path[], int16_t* &data, size_t &size, int &max, int &min){
	ifstream file(file_path);
	if (!file.is_open()) {
		cerr << "Error opening file: " << file_path << endl;
		return;
	}

	vector<int16_t> vec;
	int temp;
	while (file >> temp) {
		vec.push_back(temp);
	}
	file.close();
	size = vec.size();
	data = new int16_t[size];
	copy(vec.begin(), vec.end(), data);

	max = *max_element(data, data + size);
	min = *min_element(data, data + size);
}


// ****** 码流IO: BitInputStream 仅用作解码器
// BitInputStream::BitInputStream(istream &in) :
// 	input(in),
// 	currentByte(0),
// 	numBitsRemaining(0) {};
BitInputStream::BitInputStream(const char *binfile)
    : currentByte(0), numBitsRemaining(0), input(binfile, std::ios::binary)
{
    if (!input) {
        std::cerr << "Error opening file." << std::endl;
    }
}

int BitInputStream::read() {
	if (currentByte == char_traits<char>::eof())
		return -1;
	if (numBitsRemaining == 0) {
		currentByte = input.get();  
		if (currentByte == char_traits<char>::eof())
			return -1;
		if (!(0 <= currentByte && currentByte <= 255))
			throw logic_error("Assertion error");
		numBitsRemaining = 8;
	}
	if (numBitsRemaining <= 0)
		throw logic_error("Assertion error");
	numBitsRemaining--;
	return (currentByte >> numBitsRemaining) & 1;
}


int BitInputStream::readNoEof() {
	int result = read();
	if (result != -1)
		return result;
	else
		throw runtime_error("End of stream");
}



BitArrayInputStream::BitArrayInputStream(const uint8_t* inputData, size_t size) 
    : data(inputData), dataSize(size), currentIndex(0), currentByte(0), numBitsRemaining(0) {}

int BitArrayInputStream::read() {
    if (numBitsRemaining == 0) {
        if (currentIndex >= dataSize)
            return -1;
        currentByte = data[currentIndex++];
        numBitsRemaining = 8;
    }
    if (numBitsRemaining <= 0)
        throw std::logic_error("Assertion error");
    numBitsRemaining--;
    return (currentByte >> numBitsRemaining) & 1;
}

int BitArrayInputStream::readNoEof() {
    int result = read();
    if (result != -1)
        return result;
    else
        throw std::runtime_error("End of stream");
}


// ****** 码流IO: BitOutputStream 仅用作编码器
BitOutputStream::BitOutputStream(uint8_t* _addr) :
	currentByte(0),
	numBitsFilled(0) ,
	data_addr(_addr) {}

BitOutputStream::BitOutputStream(size_t len):
	currentByte(0),
	numBitsFilled(0) {
	data_addr = new uint8_t[len * 10];
}

void BitOutputStream::write(int b) {
	if (b != 0 && b != 1)
		throw domain_error("Argument must be 0 or 1");
	currentByte = (currentByte << 1) | b;
	numBitsFilled++;
	if (numBitsFilled == 8) {
		if (numeric_limits<char>::is_signed)
			currentByte -= (currentByte >> 7) << 8;
		data_addr[size++]=static_cast<uint8_t>(currentByte);
		currentByte = 0;
		numBitsFilled = 0;
	}
}


void BitOutputStream::finish() {
	while (numBitsFilled != 0)
		write(0);
}


bool readBinaryFile(char file_path[], void*& data, size_t& size) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return false;
    }

    // 获取文件大小
    file.seekg(0, ios::end);
    size = file.tellg();
    file.seekg(0, ios::beg);

    // 分配内存
    data = new char[size];

    // 读取文件数据到数组
    file.read(reinterpret_cast<char*>(data), size);
    file.close();

    return true;
}




// ****** GMM频率表
inline int64_t clamp(int64_t x, int64_t l, int64_t r) {
    if (x < l) return l;
    else if (x > r) return r;
    else return x;
}


int softmax(int probs[3],uint16_t scale_pred ,uint16_t exp_table[],int scale=1000,int x_bound=-12){
    int mx=*max_element(probs,probs+3);
    int sum=0;
    int base=-x_bound*scale;

    for(int i=0;i<3;i++) {
        long idx=probs[i]-mx;
        idx= max(idx*scale/scale_pred +base,0l);
        probs[i]= exp_table[idx];
        sum+=probs[i];
    }
    return sum;
}




EncTable::EncTable(uint32_t _freqs_resolution,int _low_bound,int _high_bound){
	readBinaryFile(exp_file_path, reinterpret_cast<void*&>(exp_table), exp_size);
	readBinaryFile(cdf_file_path, reinterpret_cast<void*&>(cdf_table), cdf_size);
	low_bound=_low_bound;
	high_bound=_high_bound;
	freqs_resolution=_freqs_resolution;
}


void EncTable::update(int m_probs[3],int m_means[3],int m_stds[3]){
	probs=m_probs;
	means=m_means;
	stds=m_stds;
	for(int i=0 ;i<3;i++)
		stds[i] = max(abs(stds[i]),1);
	prob_sum= softmax(m_probs,scale_pred,exp_table,exp_scale,exp_x_bound);
}


void EncTable::get_bound(int x) {
	uint32_t y_bound = UINT32_MAX;
	int base = -cdf_x_bound * cdf_scale;

	l_bound = 0, r_bound = 0;
	for (int i = 0; i < 3; i++) {
		int64_t idx_l = (long long)((low_bound * scale_pred - means[i]) - scale_pred / 2) * cdf_scale / stds[i];
		idx_l = clamp(idx_l, -base, base);
		idx_l += base;

		int64_t idx_r = (long long)((high_bound * scale_pred - means[i]) + scale_pred / 2) * cdf_scale / stds[i];
		idx_r = clamp(idx_r, -base, base);
		idx_r += base;

		l_bound += uint64_t(1) * freqs_resolution * cdf_table[idx_l] / prob_sum * probs[i];
		r_bound += uint64_t(1) * freqs_resolution * cdf_table[idx_r] / prob_sum * probs[i];
	}
	l_bound /= y_bound;  // cdf
	r_bound /= y_bound;  // cdf

	uint64_t low = 0, high = 0;
	for (int i = 0; i < 3; i++) {
		int64_t idx_l = (long long)((x * scale_pred - means[i]) - scale_pred / 2) * cdf_scale / stds[i];
		idx_l = clamp(idx_l, -base, base);
		idx_l += base;

		int64_t idx_r = (long long)((x * scale_pred - means[i]) + scale_pred / 2) * cdf_scale / stds[i];
		idx_r = clamp(idx_r, -base, base);
		idx_r += base;

		low += uint64_t(1) * freqs_resolution * cdf_table[idx_l] / prob_sum * probs[i];
		high += uint64_t(1) * freqs_resolution * cdf_table[idx_r] / prob_sum * probs[i];
	}
	low /= y_bound;  // cdf
	high /= y_bound;  // cdf

	sym_low = low - l_bound + (x - low_bound);  // (x-0.5) to (low-0.5) cumulative sum & increment for each point
	sym_high = high - l_bound + (x - low_bound + 1);  // (x+0.5) to (low-0.5) cumulative sum & increment for each point
	total_freqs = r_bound - l_bound + (high_bound - low_bound + 1); // (high+0.5) to (low-0.5) cumulative sum & increment for each point
}


EncTable::~EncTable(){
	delete[] exp_table;
	delete[] cdf_table;
}



DecTable::DecTable(int _freqs_resolution,int _low_bound,int _high_bound){
	readBinaryFile(exp_file_path, reinterpret_cast<void*&>(exp_table), exp_size);
	readBinaryFile(cdf_file_path, reinterpret_cast<void*&>(cdf_table), cdf_size);
	low_bound=_low_bound;
	high_bound=_high_bound;
	freqs_resolution=_freqs_resolution;
	sym_low=new int[high_bound-low_bound+3];
}


void DecTable::update(int m_probs[3],int m_means[3],int m_stds[3]){
	probs=m_probs;
	means=m_means;
	stds=m_stds;
	for(int i=0 ;i<3;i++)
		stds[i] = max(abs(stds[i]),1);
	prob_sum= softmax(m_probs,scale_pred,exp_table,exp_scale,exp_x_bound);
}


void DecTable::get_bounds() {
	uint32_t y_bound = UINT32_MAX;
	int base = -cdf_x_bound * cdf_scale;

	for (long long x = low_bound; x <= high_bound+1; x++) {
		uint64_t low = 0;
		for (int i = 0; i < 3; i++) {
			int64_t idx_l = (long long)((x * scale_pred - means[i]) - scale_pred / 2) * cdf_scale / stds[i];
			idx_l = clamp(idx_l, -base, base);
			idx_l += base;
			low += uint64_t(1) * freqs_resolution * cdf_table[idx_l] / prob_sum * probs[i];
		}
		low /= y_bound;  // cdf
		int idx = x - low_bound;
		sym_low[idx] = low + (x - low_bound);  // (x-0.5) to (low-0.5) cumulative sum & increment by 1 for each point
	}

	uint32_t l_cdf = sym_low[0], r_cdf = sym_low[high_bound - low_bound +1];
	for (int x = low_bound; x <= high_bound+1; x++) {
		int idx = x - low_bound;
		sym_low[idx] -= l_cdf;
	}
	total_freqs = r_cdf - l_cdf;
}


DecTable::~DecTable(){
	delete[] exp_table;
	delete[] cdf_table;
	delete[] sym_low;
}






// ****** 编码器
ArithmeticCoderBase::ArithmeticCoderBase(int numBits) {
	if (!(1 <= numBits && numBits <= 63))
		throw domain_error("State size out of range");
	numStateBits = numBits;
	fullRange = static_cast<decltype(fullRange)>(1) << numStateBits;
	halfRange = fullRange >> 1;  // Non-zero
	quarterRange = halfRange >> 1;  // Can be zero
	minimumRange = quarterRange + 2;  // At least 2
	maximumTotal = min(numeric_limits<decltype(fullRange)>::max() / fullRange, minimumRange);
	stateMask = fullRange - 1;
	low = 0;
	high = stateMask;
}

ArithmeticCoderBase::~ArithmeticCoderBase() {}

void ArithmeticCoderBase::update(uint32_t total, uint32_t symLow, uint32_t symHigh){
	uint64_t range = high - low + 1;
	uint64_t newLow  = low + symLow  * range / total;
	uint64_t newHigh = low + symHigh * range / total - 1;
	low = newLow;
	high = newHigh;

	while (((low ^ high) & halfRange) == 0) {
		int bit = static_cast<int>(low >> (numStateBits - 1));
		if(bit!=0 && bit!=1)
			throw logic_error("Assertion error");
		shift();
		low  = ((low  << 1) & stateMask);
		high = ((high << 1) & stateMask) | 1;
	}

	while ((low & ~high & quarterRange) != 0) {
		underflow();
		low = (low << 1) ^ halfRange;
		high = ((high ^ halfRange) << 1) | halfRange | 1;
	}
}

ArithmeticEncoder::ArithmeticEncoder(int numBits, size_t len) :
	ArithmeticCoderBase(numBits),
	output(len),
	numUnderflow(0) {
		bin = output.data_addr;
	}

void ArithmeticEncoder::write(uint32_t total, uint32_t symLow, uint32_t symHigh) {
	update(total, symLow, symHigh);
}

void ArithmeticEncoder::finish() {
	output.write(1);
}

void ArithmeticEncoder::shift() {
	int bit = static_cast<int>(low >> (numStateBits - 1));
	if(bit!=0 && bit!=1)
		throw logic_error("Assertion error");
	output.write(bit);
	
	for (; numUnderflow > 0; numUnderflow--)
		output.write(bit ^ 1);
}

void ArithmeticEncoder::underflow() {
	if (numUnderflow == numeric_limits<decltype(numUnderflow)>::max())
		throw overflow_error("Maximum underflow reached");
	numUnderflow++;
}




// ****** 解码器
ArithmeticDecoder::ArithmeticDecoder(int numBits, const char *binfile) :
		ArithmeticCoderBase(numBits),
		input(binfile),
		code(0) {
	for (int i = 0; i < numStateBits; i++)
		code = code << 1 | readCodeBit();
}

ArithmeticPixelDecoder::ArithmeticPixelDecoder(int numBits, const char *binfile, size_t len, int16_t gmm_scale, int16_t xmin, int16_t xmax) :
	dec(numBits, binfile),freqs_table(1000000, xmin, xmax){
	freqs_table.scale_pred = gmm_scale;
	this->len = len;
}

int ArithmeticPixelDecoder::read(vector<int>gmm){
	int* ptr = gmm.data();
    freqs_table.update(ptr+6, ptr, ptr+3);
	freqs_table.get_bounds();
	int16_t symbol = dec.read(freqs_table);
	if (++p>len)
		throw logic_error("the number of pixels exceeds the length of the array");
	return symbol;
}

int ArithmeticDecoder::read(DecTable &freqs) {
	// Translate from coding range scale to frequency table scale
	uint32_t total = freqs.total_freqs;
	// printf("total: %lld\n", total);
	// printf("maximumTotal: %lld", maximumTotal);
	if (total > maximumTotal)
		throw invalid_argument("--Cannot decode symbol because total is too large");
	uint64_t range = high - low + 1;
	uint64_t offset = code - low;// 200-100=100
	uint64_t value = ((offset + 1) * total - 1) / range;  //eg 100 100/288 A:90-101 B:102-110
	if (value * range / total > offset)
		throw logic_error("Assertion error");
	if (value >= total)
		throw logic_error("Assertion error");
	
	// A kind of binary search. Find highest symbol such that freqs.getLow(symbol) <= value.
	uint32_t start = 0;
	uint32_t end = freqs.high_bound-freqs.low_bound +1;
	while (end - start > 1) {
		uint32_t middle = (start + end) >> 1;
		if (freqs.sym_low[middle] > value)
			end = middle;
		else
			start = middle;
	}
	if (start + 1 != end)
		throw logic_error("Assertion error");
	
	int symbol = start;
	if (!(freqs.sym_low[symbol] * range / total <= offset && offset < freqs.sym_low[symbol+1] * range / total))
		throw logic_error("Assertion error");
	update(total, freqs.sym_low[symbol], freqs.sym_low[symbol+1]);
	if (!(low <= code && code <= high))
		throw logic_error("Assertion error: Code out of range");
	return symbol+freqs.low_bound;
}


void ArithmeticDecoder::shift() {
	code = ((code << 1) & stateMask) | readCodeBit();
}


void ArithmeticDecoder::underflow() {
	code = (code & halfRange) | ((code << 1) & (stateMask >> 1)) | readCodeBit();
}


int ArithmeticDecoder::readCodeBit() {
	int temp = input.read();
	if (temp == -1)
		temp = 0;
	return temp;
}


ArithmeticBlockEncoder::ArithmeticBlockEncoder(int numBits, size_t len, int16_t gmm_scale, int16_t xmin, int16_t xmax):
    enc(numBits, len), len(len), freqs_table(1000000, xmin, xmax){
    freqs_table.scale_pred = gmm_scale;
}


vector<uint8_t> ArithmeticBlockEncoder::coding(vector<vector<int>> gmm, int16_t *data)
{
    if (freqs_table.low_bound >= freqs_table.high_bound) vector<uint8_t> res;
    for (int p = 0; p < len; p++) {
		int* gmm_p = gmm[p].data();
        freqs_table.update(gmm_p+6, gmm_p, gmm_p+3);
        freqs_table.get_bound(data[p]);
        int total_freqs = freqs_table.total_freqs, symlow = freqs_table.sym_low, symhigh = freqs_table.sym_high;
        enc.write(total_freqs, symlow, symhigh);
    }
    enc.finish();
    enc.output.finish();
	vector<uint8_t> res (enc.bin, enc.bin+enc.output.size);
	free(enc.bin);
    return res;
}

vector<uint8_t> coding(vector<vector<vector<int>>>gmms, vector<vector<int16_t>>datas, vector<int16_t> gmm_scales)
{
	printf("coding\n");
	printf("gmms size: %d %d %d\n", gmms.size(), gmms[0].size(), gmms[0][0].size());
	printf("datas size: %d %d\n", datas.size(), datas[0].size());
    const int16_t ratios[13] = {16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2};
    const int blk_size = 64;
    int16_t mxs[39], mns[39];
	vector<vector<uint8_t>> blk_res;
	size_t total_len = 0;
    for (int sub = 0; sub < 13; ++sub){
        for (int ch=0;ch<3;ch++) {
            int H = (ch == 0)? 2160 : 1088;
            int W = (ch == 0)? 3840 : 1920;
            int h = H / ratios[sub];
            int w = W / ratios[sub];
            int idx = sub*3 + ch;
            vector<vector<int>>& gmm = gmms[idx];
            vector<int16_t>& data = datas[idx];
            int16_t xmin = *min_element(data.begin(), data.begin() + h*w);
            int16_t xmax = *max_element(data.begin(), data.begin() + h*w);
            mxs[idx] = xmax;
            mns[idx] = xmin;

            for (int row = 0; row < h; row += blk_size) {
                for (int col = 0; col < w; col += blk_size) {
                    int row_end = min(row + blk_size, h);
                    int col_end = min(col + blk_size, w);
                    size_t blk_len = (row_end - row) * (col_end - col);

                    vector<vector<int>> blk_gmm(blk_len);
                    int16_t* blk_data = new int16_t[blk_len];

                    for (int i = row, p = 0; i < row_end; ++i) {
                        for (int j = col; j < col_end; ++j, ++p) {
                            blk_gmm[p] = gmm[i * w + j];
                            blk_data[p] = data[i * w + j];
                        }
                    }
					ArithmeticBlockEncoder enc(32, blk_len, gmm_scales[idx], xmin, xmax);
					vector<uint8_t> res = enc.coding(blk_gmm, blk_data);
					blk_res.push_back(res);
        			total_len += res.size();
					printf("blk_res size: %d res size %d\n", blk_res.size(), res.size());
					delete[] blk_data;

                }
            }
        }
    }
    // 2. 依次编码各个块

    total_len = 39 * 4 + blk_res.size() * 2 + total_len; 
    vector<uint8_t> stream(total_len);
    size_t offset = 0; // 当前写入位置
    // 3.1 写入 min 和 max
    for (int i = 0; i < 39; i++) {
        stream[offset++] = mns[i] & 0xFF;
        stream[offset++] = (mns[i] >> 8) & 0xFF;
        stream[offset++] = mxs[i] & 0xFF;
        stream[offset++] = (mxs[i] >> 8) & 0xFF;
    }

    // 3.2 写入块长度
    for (auto& res : blk_res) {
        stream[offset++] = res.size() & 0xFF;
        stream[offset++] = (res.size() >> 8) & 0xFF;
    }

    // 3.3 写入各个块的码流
    for (auto& res : blk_res) {
        memcpy(stream.data() + offset, res.data(), res.size());
        offset += res.size();
    }
    return stream;
}
