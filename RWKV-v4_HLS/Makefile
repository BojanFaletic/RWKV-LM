CC=clang++
SANATIZERS=-fsanitize=address -fno-omit-frame-pointer

phony:
	$(CC) run.cpp -Wall -Wextra -std=c++20 -Isrc -o run $(SANATIZERS)

clean:
	rm -f run

