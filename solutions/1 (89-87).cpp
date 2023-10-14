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

int main() {
    _setmode(_fileno(stdout), 0x00020000);
    _setmode(_fileno(stdin), 0x00020000);
    _setmode(_fileno(stderr), 0x00020000);

    std::wifstream input_file(R"(C:\Projects\ML\RUCODE\D\in.txt)", std::ios::in);
    std::wofstream output_file(R"(C:\Projects\ML\RUCODE\D\stupid_answers.txt)", std::ios::out);

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
        int min_levenstein = std::numeric_limits<int>::max();
        int best_word_index = -1;

        std::wstring &word = test_words[i];

        int start_position = int(std::lower_bound(
            train_words.begin(), train_words.end(), word
        ) - train_words.begin());

        for (int j = std::max(0, start_position - K);
            j < std::min(start_position + K, (int)train_words.size()); j++) {

            int current = levenstein(word, train_words[j]);
            int stress_position = train_positions[j];
            if ((current < min_levenstein || 
                (current == min_levenstein && rnd() % 3 == 0))
                && stress_position < word.size()
                && vowels.contains(word[stress_position])) {

                best_word_index = j;
                min_levenstein = current;
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