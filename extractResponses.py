def extractResponses():
    f = open("output_regulared.txt",encoding='utf-8')
    tests = ["你/好","你/叫/什/么/名/字","你/几/岁/了","我/喜/欢/你","我/想/要/女/朋/友"]
    result = [[],[],[],[],[]]

    line = f.readline()
    while line:
        nextLine = f.readline()
        line = line.replace('\n', '')
        _nextLine = nextLine.replace('\n', '')
        _nextLine = _nextLine.split('/')
        if line == tests[0] and _nextLine!=['']:
            result[0].append(_nextLine)
        if line == tests[1] and _nextLine!=['']:
            result[1].append(_nextLine)
        if line == tests[2] and _nextLine!=['']:
            result[2].append(_nextLine)
        if line == tests[3] and _nextLine!=['']:
            result[3].append(_nextLine)
        if line == tests[4] and _nextLine!=['']:
            result[4].append(_nextLine)
        line = nextLine
    f.close()
    return result
if __name__ == '__main__':
    for res in extractResponses():
        print(res)