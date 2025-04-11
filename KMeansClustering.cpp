#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <ranges>
#include <functional>
#include <queue>
#include <fstream>
#include <chrono>
#include <limits>

using namespace std;

class KMeans
{
public:
	static enum PartitioningMethod
	{
		Random,
		RoundRobin
	};

	explicit KMeans(
		const set<vector<double>>& data,
		const int d,
		const int k,
		KMeans::PartitioningMethod method = KMeans::PartitioningMethod::Random,
		const function<double(vector<double>, vector<double>, const int)>& distanceF = nullptr)
		: k_{ k }, dim_{ d }, data_{ data }
	{
		if (distanceF == nullptr)
		{
			distanceF_ = [](const vector<double>& a, const vector<double>& b, const int dim) { return euclidean_distance(a, b, dim); };
		}
		else
		{
			distanceF_ = distanceF;
		}

		clusters_ = partition_in_k_sets(data_, k_, method);
	}


	vector<set<vector<double>>> get()
	{
		priority_queue<VectorInfo, vector<VectorInfo>, decltype(&cmpVectorInfoMax)> vector_info_;
		while (true) {
			vector<set<vector<double>>> new_clusters(k_);
			auto centroids = calculate_centroids(clusters_);
			vector_info_ = priority_queue<VectorInfo, vector<VectorInfo>, decltype(&cmpVectorInfoMax)>{ cmpVectorInfoMax };

#pragma omp parallel for shared(centroids)
			for (int i = 0; i < k_; ++i) {
				for (int j = 0; j < clusters_[i].size(); ++j) {
					auto it = clusters_[i].begin();
					std::advance(it, j);
					auto info = vector_to_centroid_info(*it, i, centroids);
#pragma omp critical 
					{
						new_clusters[info.centroid_min_index].insert(*it);
						vector_info_.push(std::move(info));
					}
				}
			}

			for (int k = 0; k < new_clusters.size(); ++k) {
				if (new_clusters[k].empty()) {
					auto fartherst = vector_info_.top();
					vector_info_.pop();

					auto it = data_.begin();
					std::advance(it, fartherst.offset);

					new_clusters[k].insert(*it);
					new_clusters[fartherst.centroid_min_index].erase(*it);
				}
			}

			if (clusters_equal(clusters_, new_clusters)) {
				break;
			}

			clusters_ = std::move(new_clusters);
		}

		return clusters_;
	}

private:
	struct VectorInfo {
		int offset;
		double centroid_min;
		int centroid_min_index;

		bool operator==(const VectorInfo& other) const
		{
			return offset == other.offset
				&& centroid_min == other.centroid_min
				&& centroid_min_index == other.centroid_min_index;
		}

		bool operator!=(const VectorInfo& other) const
		{
			return !(*this == other);
		}

	};

	bool clusters_equal(const vector<set<vector<double>>>& a, const vector<set<vector<double>>>& b) {
		if (a.size() != b.size()) return false;
		bool equal = true;

#pragma omp parallel for reduction(&&:equal) shared(a,b)
		for (int i = 0; i < k_; i++) {
			equal = equal && a[i] == b[i];
		}
		return equal;
	}

	static bool cmpVectorInfoMax(const VectorInfo& a, const VectorInfo& b)
	{
		return a.centroid_min < b.centroid_min;
	};

	int k_;
	int dim_;
	set<vector<double>> data_;
	vector<set<vector<double>>> clusters_;
	function<double(vector<double>, vector<double>, const int)> distanceF_;

	static double euclidean_distance(vector<double> a, vector<double> b, const int d)
	{
		double sum = 0;
#pragma omp parallel for reduction(+:sum) shared(a,b)
		for (int i = 0; i < d; i++)
		{
			sum += pow(a[i] - b[i], 2);
		}

		return sqrt(sum);
	}

	vector<set<vector<double>>> partition_in_k_sets(const set<vector<double>>& data, int k, PartitioningMethod partitionStrategy) {
		vector<vector<double>> copy(data.begin(), data.end());
		vector<set<vector<double>>> result(k);

		if (partitionStrategy == PartitioningMethod::Random) {

			vector<int> sizes(k, 0);
			random_device rd;
			mt19937 g(rd());
			ranges::shuffle(copy, g);
			size_t available_space = copy.size();
			for (int i = 0; i < k; ++i) {
				if (i == k - 1) {
					sizes[i] = static_cast<int>(available_space);
				}
				else {
					uniform_int_distribution<int> distribution(1, static_cast<int>(available_space) - (k - i - 1));
					sizes[i] = distribution(g);
					available_space -= sizes[i];
				}
			}


			auto it = copy.begin();
			for (int i = 0; i < k; ++i) {
				for (int j = 0; j < sizes[i]; ++j) {
					result[i].insert(*it);
					++it;
				}
			}
		}
		else if (partitionStrategy == PartitioningMethod::RoundRobin) {
			for (int i = 0; i < k; ++i) {
				for (int j = i; j < copy.size(); j += k) {
					result[i].insert(copy[j]);
				}
			}
		}
		return result;
	}

	vector<vector<double>> calculate_centroids(const vector<set<vector<double>>>& clusters) {
		vector<vector<double>> result(k_, vector<double>(dim_, 0));

#pragma omp parallel for collapse(2) shared(result, clusters)
		for (int i = 0; i < k_; ++i) {
			for (int j = 0; j < dim_; ++j) {
				result[i][j] = vector_level_avg(clusters[i].begin(), j, clusters[i].size());
			}
		}
		return result;
	}

	double vector_level_avg(set<vector<double>>::const_iterator cluster_it, const int level_idx, const size_t cluster_size) {
		double sum = 0;

#pragma omp parallel for reduction(+:sum) shared(cluster_it, level_idx)
		for (size_t k = 0; k < cluster_size; ++k) {
			auto it = cluster_it;
			std::advance(it, k);
			sum += (*it)[level_idx];
		}

		return sum / cluster_size;
	}


	VectorInfo vector_to_centroid_info(const vector<double>& v, int offset, const vector<vector<double>>& centroids) {
		VectorInfo info;
		info.offset = offset;
		double min_dist = numeric_limits<double>::max();
		int min_dist_idx = -1;
#pragma omp parallel for
		for (int i = 0; i < centroids.size(); ++i) {
			double dist = distanceF_(v, centroids[i], dim_);
#pragma omp critical
			{
				if (dist < min_dist) {
					min_dist = dist;
					min_dist_idx = i;
				}
			}
		}

		info.centroid_min = min_dist;
		info.centroid_min_index = min_dist_idx;

		return info;
	}
};

class KMeansSerial
{
public:
	static enum PartitioningMethod
	{
		Random,
		RoundRobin
	};

	explicit KMeansSerial(
		const set<vector<double>>& data,
		const int d,
		const int k,
		KMeansSerial::PartitioningMethod method = KMeansSerial::PartitioningMethod::Random,
		const function<double(vector<double>, vector<double>, const int)>& distanceF = nullptr)
		: k_{ k }, dim_{ d }, data_{ data }
	{
		if (distanceF == nullptr)
		{
			distanceF_ = [](const vector<double>& a, const vector<double>& b, const int dim) { return euclidean_distance(a, b, dim); };
		}
		else
		{
			distanceF_ = distanceF;
		}

		clusters_ = partition_in_k_sets(data_, k_, method);
	}

	vector<set<vector<double>>> get()
	{
		priority_queue<VectorInfo, vector<VectorInfo>, decltype(&cmpVectorInfoMax)> vector_info_;
		while (true) {
			vector<set<vector<double>>> new_clusters(k_);
			auto centroids = calculate_centroids(clusters_);
			vector_info_ = priority_queue<VectorInfo, vector<VectorInfo>, decltype(&cmpVectorInfoMax)>{ cmpVectorInfoMax };

			for (int i = 0; i < k_; ++i) {
				for (int j = 0; j < clusters_[i].size(); ++j) {
					auto it = clusters_[i].begin();
					std::advance(it, j);
					auto info = vector_to_centroid_info(*it, i, centroids);
					new_clusters[info.centroid_min_index].insert(*it);
					vector_info_.push(std::move(info));
				}
			}

			for (int k = 0; k < new_clusters.size(); ++k) {
				if (new_clusters[k].empty()) {
					auto fartherst = vector_info_.top();
					vector_info_.pop();

					auto it = data_.begin();
					std::advance(it, fartherst.offset);

					new_clusters[k].insert(*it);
					new_clusters[fartherst.centroid_min_index].erase(*it);
				}
			}

			if (clusters_ == new_clusters) {
				break;
			}

			clusters_ = std::move(new_clusters);
		}

		return clusters_;
	}

private:
	struct VectorInfo {
		int offset;
		double centroid_min;
		int centroid_min_index;

		bool operator==(const VectorInfo& other) const
		{
			return offset == other.offset
				&& centroid_min == other.centroid_min
				&& centroid_min_index == other.centroid_min_index;
		}

		bool operator!=(const VectorInfo& other) const
		{
			return !(*this == other);
		}

	};

	static bool cmpVectorInfoMax(const VectorInfo& a, const VectorInfo& b)
	{
		return a.centroid_min < b.centroid_min;
	};

	int k_;
	int dim_;
	set<vector<double>> data_;
	vector<set<vector<double>>> clusters_;
	function<double(vector<double>, vector<double>, const int)> distanceF_;

	static double euclidean_distance(vector<double> a, vector<double> b, const int d)
	{
		double sum = 0;
		for (int i = 0; i < d; i++)
		{
			sum += pow(a[i] - b[i], 2);
		}

		return sqrt(sum);
	}

	vector<set<vector<double>>> partition_in_k_sets(const set<vector<double>>& data, int k, PartitioningMethod partitionStrategy) {
		vector<vector<double>> copy(data.begin(), data.end());
		vector<set<vector<double>>> result(k);

		if (partitionStrategy == PartitioningMethod::Random) {

			vector<int> sizes(k, 0);
			random_device rd;
			mt19937 g(rd());
			ranges::shuffle(copy, g);
			size_t available_space = copy.size();
			for (int i = 0; i < k; ++i) {
				if (i == k - 1) {
					sizes[i] = static_cast<int>(available_space);
				}
				else {
					uniform_int_distribution<int> distribution(1, static_cast<int>(available_space) - (k - i - 1));
					sizes[i] = distribution(g);
					available_space -= sizes[i];
				}
			}


			auto it = copy.begin();
			for (int i = 0; i < k; ++i) {
				for (int j = 0; j < sizes[i]; ++j) {
					result[i].insert(*it);
					++it;
				}
			}
		}
		else if (partitionStrategy == PartitioningMethod::RoundRobin) {
			for (int i = 0; i < k; ++i) {
				for (int j = i; j < copy.size(); j += k) {
					result[i].insert(copy[j]);
				}
			}
		}
		return result;
	}

	vector<vector<double>> calculate_centroids(const vector<set<vector<double>>>& clusters) {
		vector<vector<double>> result(k_, vector<double>(dim_, 0));

		for (int i = 0; i < k_; ++i) {
			for (int j = 0; j < dim_; ++j) {
				result[i][j] = vector_level_avg(clusters[i].begin(), j, clusters[i].size());
			}
		}
		return result;
	}

	double vector_level_avg(set<vector<double>>::const_iterator cluster_it, const int level_idx, const size_t cluster_size) {
		double sum = 0;

		for (size_t k = 0; k < cluster_size; ++k) {
			auto it = cluster_it;
			std::advance(it, k);
			sum += (*it)[level_idx];
		}

		return sum / cluster_size;
	}


	VectorInfo vector_to_centroid_info(const vector<double>& v, int offset, const vector<vector<double>>& centroids) {
		VectorInfo info;
		info.offset = offset;
		double min_dist = numeric_limits<double>::max();
		int min_dist_idx = -1;
		for (int i = 0; i < centroids.size(); ++i) {
			double dist = distanceF_(v, centroids[i], dim_);
			if (dist < min_dist) {
				min_dist = dist;
				min_dist_idx = i;
			}
		}

		info.centroid_min = min_dist;
		info.centroid_min_index = min_dist_idx;

		return info;
	}
};

double manhattan_dist(vector<double> a, vector<double> b, const int d)
{
	double sum = 0;
#pragma omp parallel for reduction(+:sum) shared(a,b)
	for (int i = 0; i < d; i++)
	{
		sum += abs(a[i] - b[i]);
	}

	return sqrt(sum);
}



int main()
{
	int N;
	int d, k;

	set<vector<double>> input;
	ifstream inFile("kmeans_input.txt");
	inFile >> N >> d >> k;

	for (int i = 0; i < N; ++i)
	{
		vector<double> vec(d);
		for (int j = 0; j < d; ++j)
		{
			inFile >> vec[j];
		}
		input.insert(std::move(vec));
	}

	cout << "Processed inputs" << endl;


	auto start = chrono::high_resolution_clock::now();
	KMeans algo{ input, d, k, KMeans::PartitioningMethod::RoundRobin };

	auto r = algo.get();

	auto end = chrono::high_resolution_clock::now();
	cout << "Time taken parallel: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

	start = chrono::high_resolution_clock::now();
	KMeansSerial algoSerial{ input, d, k, KMeansSerial::PartitioningMethod::RoundRobin };
	auto rSerial = algoSerial.get();
	end = chrono::high_resolution_clock::now();
	cout << "Time taken serial: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

	ofstream outFile("kmeans_results.txt");
	if (!outFile.is_open()) {
		cerr << "Failed to open output file" << endl;
		return 1;
	}

	int j = 1;
	for (auto& s : r)
	{
		for (auto& v : s)
		{
			outFile << j++;
			for (int i = 0; i < v.size(); ++i) {
				outFile << "," << v[i];
			}
			outFile << endl;
		}
		outFile << endl;
	}

	outFile.close();
	cout << "Results written to kmeans_results.txt" << endl;
	cin.get();

	return 0;
}