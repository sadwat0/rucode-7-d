#include <iostream>
#include <vector>
#include <set>
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#include <fstream>
#include <random>
#include <chrono>

constexpr int K = 30;

std::mt19937 rnd(1337);

std::vector<std::vector<int>> dp(50, std::vector<int>(50, 0));
int levenstein(std::wstring &a, std::wstring &b) {
    size_t n = a.size(), m = b.size();

    for (int j = 1; j <= m; j++)
        dp[0][j] = dp[0][j - 1] + 1;

    for (int i = 1; i <= n; i++) {
        dp[i][0] = dp[i - 1][0] + 1;
        for (int j = 1; j <= m; j++) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min({
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                    }) + 1;
            }
        }
    }

    return dp[n][m];
}

float jw_distance(std::wstring s1, std::wstring s2, bool caseSensitive = true) {
    float m = 0;
    int low, high, range;
    int k = 0, numTrans = 0;

    // Exit early if either are empty
    if (s1.length() == 0 || s2.length() == 0) {
        return 0;
    }

    // Convert to lower if case-sensitive is false
    if (caseSensitive == false) {
        transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
        transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
    }

    // Exit early if they're an exact match.
    if (s1 == s2) {
        return 1;
    }

    range = (std::max(s1.length(), s2.length()) / 2) - 1;
    std::vector<int> s1Matches(s1.length());
    std::vector<int> s2Matches(s2.length());

    for (int i = 0; i < s1.length(); i++) {

        // Low Value;
        if (i >= range) {
            low = i - range;
        } else {
            low = 0;
        }

        // High Value;
        if (i + range <= (s2.length() - 1)) {
            high = i + range;
        } else {
            high = s2.length() - 1;
        }

        for (int j = low; j <= high; j++) {
            if (s1Matches[i] != 1 && s2Matches[j] != 1 && s1[i] == s2[j]) {
                m += 1;
                s1Matches[i] = 1;
                s2Matches[j] = 1;
                break;
            }
        }
    }

    // Exit early if no matches were found
    if (m == 0) {
        return 0;
    }

    // Count the transpositions.
    for (int i = 0; i < s1.length(); i++) {
        if (s1Matches[i] == 1) {
            int j;
            for (j = k; j < s2.length(); j++) {
                if (s2Matches[j] == 1) {
                    k = j + 1;
                    break;
                }
            }

            if (s1[i] != s2[j]) {
                numTrans += 1;
            }
        }
    }

    float weight = (m / s1.length() + m / s2.length() + (m - (numTrans / 2)) / m) / 3;
    float l = 0;
    float p = 0.1;
    if (weight > 0.7) {
        while (s1[l] == s2[l] && l < 4) {
            l += 1;
        }

        weight += l * p * (1 - weight);
    }
    return weight;
}

int main() {
    _setmode(_fileno(stdout), 0x00020000);
    _setmode(_fileno(stdin), 0x00020000);
    _setmode(_fileno(stderr), 0x00020000);

    std::wifstream input_file(R"(C:\Projects\ML\RUCODE\D\in.txt)", std::ios::in);
    std::wofstream output_file(R"(C:\Projects\ML\RUCODE\D\jw_distance.txt)", std::ios::out);

    std::locale loc("ru_RU.UTF-8");
    input_file.imbue(loc);
    output_file.imbue(loc);

    const int TRAIN_SIZE = 588490;
    const int TEST_SIZE = 294253;

    std::vector<std::wstring> train_strings, train_words;
    std::vector<int> train_positions;
    std::vector<std::wstring> test_words;

    std::wcout << "Running.\n";

    // Reading data
    std::wstring read_word;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        input_file >> read_word;
        //std::wcout << read_word << '\n';
        train_strings.push_back(read_word);

        auto accent = find(read_word.begin(), read_word.end(), L'^');
        train_positions.push_back(static_cast<int>(accent - read_word.begin() - 1));

        read_word.erase(accent);
        train_words.push_back(read_word);

        if (i % 1000 == 0) {
            std::wcout << "Read train: " << i << "/" << TRAIN_SIZE << "\n";
        }
    }

    for (int i = 0; i < TEST_SIZE; i++) {
        input_file >> read_word;
        test_words.push_back(read_word);

        if (i % 1000 == 0) {
            std::wcout << "Read test: " << i << "/" << TEST_SIZE << "\n";
        }
    }

    std::wcout << "Size of train: " << train_words.size() <<
        " | test: " << test_words.size() << "\n";

    std::set<wchar_t> vowels;
    for (wchar_t c : L"аеиоуэюяыё")
        vowels.insert(c);

    // Stupid solution
    std::vector<int> res(test_words.size(), -1);
    for (int i = 0; i < res.size(); i++) {
        float best_value = 1.0f;
        int best_word_index = -1;

        std::wstring &word = test_words[i];

        int start_position = int(std::lower_bound(
            train_words.begin(), train_words.end(), word
        ) - train_words.begin());

        for (int j = std::max(0, start_position - K);
            j < std::min(start_position + K, (int)train_words.size()); j++) {

            //int current_value = 2 * levenstein(word, train_words[j]);
            float current_value = -jw_distance(word, train_words[j]);

            int stress_position = train_positions[j];
            if ((current_value < best_value || 
                (current_value <= best_value + 1e-6 && rnd() % 3 == 0))
                && stress_position < word.size()
                && vowels.contains(word[stress_position])) {

                best_word_index = j;
                best_value = current_value;
            }
        }

        res[i] = best_word_index == -1 ? 0 : train_positions[best_word_index];
        while (res[i] < word.size() && !vowels.contains(word[res[i]]))
            res[i]++;
        while (res[i] > 0 && !vowels.contains(word[res[i]]))
            res[i]--;

        if (i % 1000 == 0) {
            std::wcout << "Completed: " << i << "/" << res.size() << "\n";
        }
    }

    std::wcout << "Done!" << "\n";

    for (int i = 0; i < res.size(); i++) {
        if (res[i] == -1 || res[i] >= test_words[i].size())
            std::wcout << "Error!\n";

        output_file << test_words[i].substr(0, res[i] + 1) <<
            L"^" << test_words[i].substr(res[i] + 1) << "\n";
    }


    return 0;
}