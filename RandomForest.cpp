#include<iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;

struct Node {
	vector<Node*> child;
	map<string, vector<string>> data;
	vector<int> label;
	string feature;
	string value;
	int ans = -1;
	bool isLeaf;
	Node(map<string, vector<string>> data, vector<int> label, string feature, string value, bool isLeaf) :
		data(data), label(label), feature(feature), value(value), isLeaf(isLeaf) {}
};

struct Tree {
	Node* root = nullptr;
	Tree(map<string, vector<string>> data, vector<int> label);
	map<string, vector<double>> calculateEnt(map<string, vector<string>>& data, vector<int>& label);
	map<string, double> calculateGain(map<string, vector<double>> Ent);
	string findMax(map<string, double> Gain);
	void createTree(Node* current);
	int predict(map<string, string> que);
};

double k(double p) {
	return -(log2(p) * p);
}

Tree::Tree(map<string, vector<string>> data, vector<int> label) {
	root = new Node(data, label, "", "", false);
	createTree(root);
}

map<string, vector<double>> Tree::calculateEnt(map<string, vector<string>>& data, vector<int>& label) {
	map<string, vector<double>> Ent;
	int t = 0;
	for (int i = 0; i < label.size(); i++) {
		if (label[i] == 0) t += 1;
	}
	Ent["label"].push_back(k(t / (double)label.size()) + k(1 - t / (double)label.size()));
	for (auto it : data) {
		map<string, int> res1;
		map<string, int> res2;
		for (int i = 0; i < it.second.size(); i++) {
			res2[it.second[i]]++;
			if (label[i] == 0)res1[it.second[i]]++;
		}
		vector<double> entn;
		for (auto i : res2) {
			double qqq = res1[i.first] / (double)res2[i.first];
			entn.push_back(res2[i.first] / (double)it.second.size() * ((qqq==0 || qqq==1)?0:(k(qqq) + k(1 - qqq))));
		}
		Ent[it.first] = entn;
	}
	return Ent;
}

map<string, double> Tree::calculateGain(map<string, vector<double>> Ent) {
	map<string, double> Gain;
	for (auto it : Ent) {
		Gain[it.first] = Ent["label"][0];
		for (int i = 0; i < it.second.size(); i++) {
			Gain[it.first] -= Ent[it.first][i];
		}
	}
	Gain.erase("label");
	return Gain;
}

string Tree::findMax(map<string, double> Gain) {
	string max;
	double num = -1;
	for (auto it : Gain) {
		if (it.second > num) {
			num = it.second;
			max = it.first;
		}
	}
	return max;
}

void Tree::createTree(Node* current) {
	bool end = true;
	int start = current->label[0];
	int num0 = 0;
	int num1 = 0;
	for (int i = 0; i < current->label.size(); i++) {
		if (current->label[i] != start) end = false;
		if (current->label[i] == 0) num0++;
		else num1++;
	}
	if (end) {
		current->isLeaf = true;
		current->ans = start;
		return;
	}
	current->ans = num0 > num1 ? 0 : 1;
	if (current->isLeaf) {
		return;
	}
	map<string, vector<string>> data = current->data;
	vector<int> label = current->label;
	string max = findMax(calculateGain(calculateEnt(data, label)));
	string feature = max;
	map<string, vector<int>> dis;
	for (int i = 0; i < data[max].size(); i++) {
		dis[data[max][i]].push_back(i);
	}
	for (auto it : dis) {
		vector<int> label_;
		for (int i = 0; i < it.second.size(); i++) {
			label_.push_back(label[it.second[i]]);
		}
		map<string, vector<string>> data_;
		for (auto t : data) {
			for (int i = 0; i < it.second.size(); i++) {
				data_[t.first].push_back(t.second[it.second[i]]);
			}
		}
		data_.erase(max);
		Node* node = new Node(data_, label_, max, it.first, false);
		if (node->data.size() == 0) {
			node->isLeaf = true;
		}
		current->child.push_back(node);
		createTree(node);
	}
}

int Tree::predict(map<string, string> que) {
	Node* current = root;
	while (!current->isLeaf) {
		bool isFind = false;
		for (int j = 0; j < current->child.size(); j++) {
			if (que[current->child[j]->feature] == current->child[j]->value) {
				current = current->child[j];
				isFind = true;
				break;
			}
		}
		if (!isFind) {
			break;
		}
	}
	return current->ans;
}


struct RandomForest {
	int maxTree;
	int maxFeature;
	vector<Tree> trees;
	RandomForest(int maxTree, int maxFeature) :
		maxTree(maxTree), maxFeature(maxFeature) {
		random_device rd;
		mt19937 key(rd());
		srand(key());
	}
	vector<string> sampleFeatures(map<string, vector<string>> data);
	map<string, vector<string>> sampleData(map<string, vector<string>> data, vector<int> label, vector<int>& sampleLabel);
	void fit(map<string, vector<string>> data, vector<int> label);
	int predict(map<string, string> que);
};

vector<string> RandomForest::sampleFeatures(map<string, vector<string>> data) {
	vector<string> features;
	for (auto feature : data) {
		features.push_back(feature.first);
	}
	random_shuffle(features.begin(), features.end());
	features.resize(maxFeature);
	return features;
}

map<string, vector<string>> RandomForest::sampleData(map<string, vector<string>> data, vector<int> label, vector<int>& sampledLabel) {
	map<string, vector<string>> sampleData;
	for (auto feature : data) {
		sampleData[feature.first] = vector<string>(label.size());
	}
	for (int i = 0; i < label.size(); i++) {
		int idx = rand() % label.size();
		sampledLabel.push_back(label[idx]);
		for (auto feature : data) {
			sampleData[feature.first][i] = feature.second[idx];
		}
	}
	return sampleData;
}

void RandomForest::fit(map<string, vector<string>> data, vector<int> label) {
	for (int i = 0; i < maxTree; i++) {
		vector<int> sampledLabel;
		map<string, vector<string>> sampledData = sampleData(data, label, sampledLabel);
		vector<string> sampledFeatures = sampleFeatures(data);
		map<string, vector<string>> reducedData;
		for (auto feature : sampledFeatures) {
			reducedData[feature] = sampledData[feature];
		}
		trees.emplace_back(reducedData, sampledLabel);
	}
}

int RandomForest::predict(map<string, string> que) {
	map<int, int> votes;
	for (auto tree : trees) {
		int pre = tree.predict(que);
		votes[pre]++;
	}
	int max = 0;
	int answer = -1;
	for (auto vote : votes) {
		if (vote.second > max) {
			max = vote.second;
			answer = vote.first;
		}
	}
	return answer;
}

void loadAll(map<string, vector<string>>& data, map<string, vector<string>>& test, vector<int>& label, vector<int>& testLabel, string fileaddress, double p) {
	random_device rd;
	mt19937 key(rd());
	bernoulli_distribution dis(p);
	ifstream file(fileaddress);
	if (!file.is_open()) {
		cerr << "Error opening file" << endl;
		return;
	}
	string line;
	vector<string> headers;
	bool isFirstLine = true;
	while (getline(file, line)) {
		stringstream ss(line);
		string value;
		vector<string> row;
		while (getline(ss, value, ',')) {
			row.push_back(value);
		}
		if (isFirstLine) {
			headers = row;
			headers.pop_back(); 
			for (const auto& header : headers) {
				data[header] = vector<string>();
			}
			isFirstLine = false;
		}
		else {
			if (dis(key)) {
				for (size_t i = 0; i < row.size() - 1; ++i) {
					test[headers[i]].push_back(row[i]);
				}
				testLabel.push_back(stoi(row.back()));
			}
			else {
				for (size_t i = 0; i < row.size() - 1; ++i) {
					data[headers[i]].push_back(row[i]);
				}
				label.push_back(stoi(row.back()));
			}
		}
	}
	file.close();
}

int main() {
	double p=0.1;
	map<string, vector<string>> data,test;
	vector<int> label,testLabel;
	string fileaddress = "./1.csv";
	loadAll(data, test, label, testLabel, fileaddress, p);
	RandomForest rf(20,3);
	rf.fit(data, label);
	int correct = 0;
	for (int i = 0; i < testLabel.size(); i++) {
		map<string, string> que;
		for (auto feature : test) {
			que[feature.first] = feature.second[i];
		}
		int result = rf.predict(que);
		if (result == testLabel[i]) {
			correct++;
		}
	}
	double accuracy = correct / (double)testLabel.size();
	cout << "Accuracy: " << accuracy << endl;
	return 0;
}