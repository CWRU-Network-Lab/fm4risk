# data saving file by nathaniel hahn

import os
import uuid

dataPath = './data/'

# build data repo
if os.path.exists(dataPath):
    ## place in prompt dir
    pass
else:
    os.mkdir("./data")


# build directory off of keyword
def makePromptDir(keyword):
    if os.path.exists(dataPath+keyword):
        return dataPath + keyword
    else:
        path = os.path.join(dataPath, keyword)
        os.mkdir(path)
        return dataPath + keyword

# generate random hash as name and save relevant data
def exportAsText(results, scores, prompt, query, keyword):

    keywordPath = makePromptDir(keyword)

    filename = str(uuid.uuid4())
    filePath = os.path.join(keywordPath, filename)
    score_str = " ".join(str(score) for score in scores)

    with open(filePath, "x") as resultFile:
        resultFile.write("%s \n %s \n %s \n %s \n" % (query, prompt, score_str, results))
        resultFile.close()

    print("data saved in: " + filePath)



