# runner file written by nathaniel hahn

import opto_gpu
import caller
import pickle
import os
import saver


retry_prompt = ""


## eval test set
function_requests = [
"Write a function to calculate the factorial of a number.",
"Write a function to check if a number is prime.",
"Write a function to find the greatest common divisor (GCD) of two numbers.",
"Write a function to compute the nth Fibonacci number.",
"Write a function to reverse a string.",
"Write a function to check if a string is a palindrome.",
"Write a function to find the largest element in a list.",
"Write a function to remove duplicates from a list.",
"Write a function to merge two sorted lists into a single sorted list.",
"Write a function to calculate the sum of all elements in a list.",
"Write a function to find the second largest element in a list.",
"Write a function to sort a list using bubble sort.",
"Write a function to convert a decimal number to binary.",
"Write a function to count the number of vowels in a string.",
"Write a function to find the length of a list.",
"Write a function to check if a number is even or odd.",
"Write a function to calculate the power of a number.",
"Write a function to get the HTML of a website.",
"Write a function to read a file and return its contents.",
"Write a function to write a string to a file.",
"Write a function to find the minimum element in a list.",
"Write a function to compute the square root of a number using the Newton-Raphson method.",
"Write a function to flatten a nested list.",
"Write a function to find the intersection of two lists.",
"Write a function to find the union of two lists.",
"Write a function to calculate the average of a list of numbers.",
"Write a function to rotate a list by a given number of positions.",
"Write a function to implement a binary search on a sorted list.",
"Write a function to check if a string is an anagram of another.",
"Write a function to find the longest common prefix among a list of strings.",
"Write a function to generate all permutations of a string.",
"Write a function to find the first non-repeating character in a string.",
"Write a function to implement the quicksort algorithm.",
"Write a function to calculate the area of a circle given its radius.",
"Write a function to find the transpose of a matrix.",
"Write a function to implement a queue using two stacks.",
"Write a function to count the frequency of each element in a list.",
"Write a function to generate a random password with given constraints.",
"Write a function to check if a given year is a leap year.",
"Write a function to convert a string to title case.",
"Write a function to convert a binary number to decimal.",
"Write a function to find the longest palindrome substring in a string.",
"Write a function to find the shortest path in a graph using Dijkstra's algorithm.",
"Write a function to implement a stack using a linked list.",
"Write a function to convert an infix expression to postfix notation.",
"Write a function to evaluate a postfix expression.",
"Write a function to detect a cycle in a directed graph.",
"Write a function to calculate the determinant of a matrix.",
"Write a function to implement depth-first search (DFS) in a graph.",
"Write a function to implement breadth-first search (BFS) in a graph.",
"Write a function to calculate the hamming distance between two strings.",
"Write a function to find the largest prime factor of a number.",
"Write a function to implement the merge sort algorithm.",
"Write a function to solve the Tower of Hanoi problem.",
"Write a function to compute the dot product of two vectors.",
"Write a function to generate the first n prime numbers.",
"Write a function to perform a left rotation on an array.",
"Write a function to check if a string contains only unique characters.",
"Write a function to find the median of a list of numbers.",
"Write a function to implement a trie data structure.",
"Write a function to find the longest increasing subsequence in a list.",
"Write a function to solve the N-Queens problem.",
"Write a function to implement a min-heap data structure.",
"Write a function to find the maximum subarray sum (Kadane's algorithm).",
"Write a function to perform matrix multiplication.",
"Write a function to implement the Sieve of Eratosthenes for prime number generation.",
"Write a function to find the lowest common ancestor in a binary tree.",
"Write a function to implement a bloom filter data structure.",
"Write a function to implement the A* search algorithm.",
"Write a function to find the maximum flow in a flow network using the Ford-Fulkerson algorithm.",
"Write a function to implement the Bellman-Ford algorithm for single-source shortest paths.",
"Write a function to perform topological sorting on a directed acyclic graph (DAG).",
"Write a function to implement the Rabin-Karp string matching algorithm.",
"Write a function to find the strongly connected components in a directed graph.",
"Write a function to implement the Huffman coding algorithm for data compression.",
"Write a function to solve the Traveling Salesman Problem using dynamic programming.",
"Write a function to implement the Boyer-Moore string search algorithm.",
"Write a function to find the maximum bipartite matching using the Hungarian algorithm.",
"Write a function to solve the Longest Common Subsequence problem using dynamic programming.",
"Write a function to implement the Red-Black Tree data structure.",
"Write a function to find the shortest supersequence of two strings.",
"Write a function to implement the Kruskal's algorithm for minimum spanning tree.",
"Write a function to solve the Edit Distance problem using dynamic programming.",
"Write a function to implement the Z-algorithm for pattern matching.",
"Write a function to find the maximum sum rectangle in a 2D matrix.",
"Write a function to implement the Johnson's algorithm for all-pairs shortest paths.",
"Write a function to solve the Coin Change problem using dynamic programming.",
"Write a function to implement the Aho-Corasick algorithm for multi-pattern string matching.",
"Write a function to find the Eulerian path in a graph.",
"Write a function to implement the Manacher's algorithm for finding the longest palindromic substring.",
"Write a function to implement tim sort",
"Write a function to square a number",
"Write a function to cube root a number",
"Write a function to determine the largest float in a list",
"Write a function to conver int to hex",
"Write a function to create a list of length n",
"Write a function to hash a string",
"Write a function to calculate the std of a list",
"Write a function to read in a text file",
"Write a function to test if a website is up",
"Write a function that capitalizes vowels in a string",
"Write a function that removes white spaces from a string",
"Write a function that counts white space in a string",
"Write a function that determine the perimeter of a trapezoid",
"Write a function to determine the area of an ellipse",
"Write a function to binary sort a list",
"Write a function to bisect a list"
]


# save these as pickles
# adjust based on model -- add command line parser
keyword = "gem_Test"
model_name = "gemini"
## current system retry prompt 
retry = "the code block created did not score well please attempt to boost the ranking by adjusting the comment/code, provide no explanation just a JSON code comment pair wrapped in curly braces, similarity score:"


# bland  -- base prompt
# salt   -- adding a random extra line
# pepper -- prompt is attempting to return something that is in correct
# sand -- attempting to get better 5, passing in 5
# gravel - attempting to get better 10
# dirt -- attempting to get better 5, passing in 10
# bedrock -- base testing
# sandstone -- messaging update

if os.path.exists("data_full.pkl") and os.path.exists("tokenized_full.pkl"):
    with open("data_full.pkl", "rb") as file:
        docs = pickle.load(file)
    with open("tokenized_full.pkl", "rb") as file:
        ds = pickle.load(file)
else:
    docs = opto_gpu.get_data()
    with open("data_full.pkl", "wb") as file:
        pickle.dump(docs, file)
    ds = opto_gpu.tokenize_ds(docs)
    with open("tokenized_full.pkl", "wb") as file:
        pickle.dump(ds, file)

## main running loop
for query in function_requests:
    try:
        scores = []
        searched = opto_gpu.search(ds, query)
        results = opto_gpu.results(searched, query, docs, 5)

        ## next we pass the results to an llm
        caller.build_convo(query, False, model_name) 
        message = caller.send_message(model_name)
        # message = testResponse
        if model == "gemini":
            newSnippet = caller.parse_message(message)
        else:
            newSnippet = caller.parse_message(message.content)
        print(newSnippet[0])
        print(newSnippet[1])
        caller.build_convo(newSnippet, True, model_name)



        for item in results:
            print(item[2])

        similarity = opto_gpu.makeCodeDoc(newSnippet[1], newSnippet[0], query)
        prompt = caller.get_prompt()
        score = opto_gpu.getRank(searched, similarity)
        print("Score: " + str(score))
        scores.append(score)   # TODO add swap for similarity[0][0].cpu().detach().numpy()[0])


        for i in range(5):
            retry_prompt =  str(str(retry) + str(similarity))
            caller.build_convo(retry_prompt, False, model_name)

            print(retry_prompt)

            retry_message = caller.send_message(model_name)
            print(retry_message)
            retry_snippet = caller.parse_message(retry_message.content)
            caller.build_convo(retry_snippet, True, model_name)

            retry_similarity = opto_gpu.makeCodeDoc(retry_snippet[1], retry_snippet[0], query)
            new_score = opto_gpu.getRank(searched, retry_similarity)
            scores.append(new_score) #similarity[0][0].cpu().detach().numpy()[0])

            print("score " + str(new_score))

            score = new_score
            similarity = retry_similarity

        print(scores)

        # save data here
        saver.exportAsText(caller.get_convo(), scores, prompt, query, keyword)
        caller.clear_convo()
        print(caller.get_convo())
    except Exception as e:
        caller.clear_convo()
        print(caller.get_convo())
        print(e)
        pass

