#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <ranges>

using namespace std;

vector<vector<int>> partition_in_k_sets(const set<int>& data, int k) {
    vector<int> copy(data.begin(), data.end());
    vector<int> sizes(k, 0);
    vector<vector<int>> result(k);

    random_device rd;
    mt19937 g(rd());
    shuffle(copy.begin(), copy.end(), g);

    int availableSpace = copy.size();

    for (int i = 0; i < k; ++i) {
        if (i == k - 1) {
            sizes[i] = availableSpace;
        }
        else {
            uniform_int_distribution<int> distribution(1, availableSpace - (k - i - 1));
            sizes[i] = distribution(g);
            availableSpace -= sizes[i];
        }
    }

    auto it = copy.begin();
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < sizes[i]; ++j) {
            result[i].push_back(*it);
            ++it;
        }
    }

    return result;
}

template<typename T>
pair<pair<T, int>, pair<T, int>> get_min_max_index(const vector<T>& vec) {
    T minValue = vec[0];
    T maxValue = vec[0];

    int minIndex = 0;
    int maxIndex = 0;

    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] < minValue) {
            minValue = vec[i];
            minIndex = i;
        }
        if (vec[i] > maxValue) {
            maxValue = vec[i];
            maxIndex = i;
        }
    }
    return { {minValue, minIndex}, {maxValue, maxIndex} };
}

double get_centroid(const vector<int>& cluster) {
    double sum = 0;
    for (int value : cluster) {
        sum += value;
    }
    return sum / cluster.size();
}

vector<vector<int>> algo(const set<int>& data, int k) {
    if (data.empty() || k <= 0 || k > data.size()) {
        return {};
    }

    vector<vector<int>> clusters = partition_in_k_sets(data, k);
    vector<double> centroids(clusters.size());

    bool changed = true;
    while (changed) {
        // Calculate centroids
        for (int i = 0; i < clusters.size(); ++i) {
            if (!clusters[i].empty()) {
                centroids[i] = get_centroid(clusters[i]);
            }
        }

        // Create new clusters based on closest centroid
        vector<vector<int>> new_clusters(k);
        for (const auto& point : data) {
            double minDistance = numeric_limits<double>::max();
            int closestCluster = 0;

            for (int i = 0; i < k; ++i) {
                if (!clusters[i].empty()) {  // Only consider non-empty clusters
                    double distance = abs(centroids[i] - point);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCluster = i;
                    }
                }
            }
            new_clusters[closestCluster].push_back(point);
        }

        // Handle empty clusters
        for (int i = 0; i < k; ++i) {
            if (new_clusters[i].empty()) {
                // Find cluster with most points
                int largestClusterIdx = 0;
                size_t maxSize = 0;

                for (int j = 0; j < k; ++j) {
                    if (new_clusters[j].size() > maxSize) {
                        maxSize = new_clusters[j].size();
                        largestClusterIdx = j;
                    }
                }

                if (maxSize > 1) {  // Only steal if the largest cluster has more than one point
                    // Find point in largest cluster that's furthest from its centroid
                    vector<double> distances;
                    for (const auto& point : new_clusters[largestClusterIdx]) {
                        distances.push_back(abs(centroids[largestClusterIdx] - point));
                    }

                    auto [minDist, maxDist] = get_min_max_index(distances);
                    int furthestPointIdx = maxDist.second;
                    int furthestPoint = new_clusters[largestClusterIdx][furthestPointIdx];

                    // Move the furthest point to the empty cluster
                    new_clusters[i].push_back(furthestPoint);
                    new_clusters[largestClusterIdx].erase(
                        new_clusters[largestClusterIdx].begin() + furthestPointIdx
                    );
                }
            }
        }

        // Check for convergence
        changed = (new_clusters != clusters);
        clusters = new_clusters;
    }

    return clusters;
}


int main()
{
    set<int> initialData = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int k = 5;

    auto res = algo(initialData, k);

    for (int i = 0; i < res.size(); ++i) {
        cout << "Cluster " << i + 1 << ": ";
        for (int j = 0; j < res[i].size(); ++j) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
