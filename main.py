from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import random
import collections

#initiate variables values
N = 4
pc = 1
pm = 0.3
k = 2 #number of bit in bin number
values = [0, 0, 0]

#declate our function. if we find min -> change sign

def f(x, n = 1):
    return n * 8 * x - n * 6 * (x**2) - n * 5 * (x**3) + n * 2 * (x**4)

#convert to Bin
def convToBin(x):
    value = ["0" for i in range(k)]
    i = 0
    while(x > 0):
        value[i] = str(x % 2)
        x //= 2
        i+=1
    value = value[::-1]
    result = ""
    for j in value:
        result += j
    return result

#convert from bin to Gray's code
def convertToGray(x):
    value = x[0]
    for i in range(0, len(x)-1):
        a = int(x[i])
        b = int(x[i+1])
        c = a ^ b
        value += str(c)
    return value

#convert from to bin from Gray's code
def convertToBinFromGray(x):
    value = x[0]
    for i in range(0, len(x) - 1):
        a = int(value[i])
        b = int(x[i + 1])
        c = a ^ b
        value += str(c)
    return value

#draw graph of function
def draw_graph(rangex = [-10, 10], rangey =[-10, 10], name="graph",title="function", xlabel="x", ylabel="y", points_x=[]):
    plt.ion()
    x = np.arange(rangex[0], rangey[1] + 0.1, 0.1)
    y = np.array(f(x))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([rangex[0], rangex[1], rangey[0], rangey[1]])

    if points_x != []:
        points_y = np.array(f(points_x))
        plt.scatter(points_x, points_y, color="red")
        plt.savefig(name + ".png")
    plt.ioff()
    plt.show()

#create first table with first parents
def tableOne(type='bin'):
    index = np.arange(1, N + 1)
    fenotype = np.arange(0, N)
    genotype = 0
    if type.lower() == 'gray':
        genotype = np.array([convertToGray(convToBin(i)) for i in fenotype])
    else :
        genotype = np.array([convToBin(i) for i in fenotype])
    koef = np.array([ f(x, -1) for x in fenotype])
    sumKoef = sum(koef)

    probability = koef / sumKoef
    sumProb = sum(probability)

    expectedNumber = probability * N
    sumExpected = sum(expectedNumber)

    realResult = np.array([round(x) for x in expectedNumber])
    sumRes = sum(realResult)

    file = open("Primary_population.txt", "w")
    file.write(u"Сума коефіцієнтів пристосованості: {}\n".format(sumKoef))
    file.write(u"Сума ймовірностей: {}\n".format(sumProb))
    file.write(u"Сума очікуваних кількостей: {}\n".format(sumExpected))
    file.write(u"Сума реально відібраних значень: {}\n\n".format(sumRes))
    values[0] = sumKoef / float(N)
    values[2] = max(koef)
    file.write(u"Середнє значення коефіцієнтів пристосованості: {}\n".format(values[0]))
    file.write(u"Максимальне значення коефіцієнта пристосованості: {}".format(values[2]))
    file.close()

    data = np.array([index, genotype, fenotype, koef, probability, expectedNumber, realResult])
    draw_graph(rangex=[-2, 4], rangey=[-20, 30], name="primary_population", title="f(x) = 8x - 6x^2 - 5x^3 + 2x^4", points_x=fenotype)
    data = data.transpose()
    names = [u"Номер особини в популяції і",
                                        u"Хромосома Xi",
                                        u"Фенотип",
                                        u"Коефіцієнт пристосованості f(Xi)",
                                        u"Ймовірність відбору в батьківський пул",
                                        u"Очікувана кількість в батьківському пулі",
                                        u"Реально відібрана кількість у батьківський пул"
                                        ]
    table = pd.DataFrame(data, columns=list(names))
    table.to_csv("primary_population.csv", index=False)
    tableTwo(data)

#make crossover for our fenotypes
def makeCross(str1, str2, point):
    point = int(point)
    a1 = str1[:point]
    b1 = str1[point:]
    a2 = str2[:point]
    b2 = str2[point:]
    return np.array([a1 + b2, a2 + b1])

#table with crossover
def tableTwo(data):
    choose = 0
    for i in range(N):
        if float(data[i][6]) >= pc:
            if choose == 0:
                choose = np.array([data[i]])
            else :
                choose = np.concatenate((choose, np.array([data[i]])), axis=0)
    indexOne = np.array([choose[x][0] for x in range(choose.shape[0])])
    fenotypeOne = np.array([choose[x][1] for x in range(choose.shape[0])])
    indexTwo = None
    fenotypeTwo = None
    for i in range(choose.shape[0]):
        if indexTwo == None and fenotypeTwo == None:
            if i + 1 <= choose.shape[0] - 1:
                indexTwo = np.array([choose[i+1][0]])
                fenotypeTwo = np.array([choose[i+1][1]])
            else:
                indexTwo = np.array([choose[choose.shape[0] - 1][0]])
                fenotypeTwo = np.array([choose[choose.shape[0] - 1][1]])
        else :
            if i + 1 <= choose.shape[0] - 1:
                indexTwo = np.concatenate((indexTwo, np.array([choose[i+1][0]])), axis=0)
                fenotypeTwo = np.concatenate((fenotypeTwo, np.array([choose[i+1][1]])), axis=0)
            else:
                l = choose.shape[0] - 1
                indexTwo = np.concatenate((indexTwo, np.array([choose[l][0]])), axis=0)
                fenotypeTwo = np.concatenate((fenotypeTwo, np.array([choose[l][1]])), axis=0)
    pointOfCross = None
    for i in range(choose.shape[0]):
        x = str(random.randint(1, k - 1))
        if pointOfCross == None:
            pointOfCross = np.array([x])
        else:
            pointOfCross = np.concatenate((pointOfCross, np.array([x])))
    offsprings = None
    offspringsOne = None
    offspringsTwo = None
    for i in range(choose.shape[0]):
        if offsprings == None:
            offsprings = makeCross(fenotypeOne[i], fenotypeTwo[i], pointOfCross[i])
        else:
            offsprings = np.concatenate((offsprings, makeCross(fenotypeOne[i], fenotypeTwo[i], pointOfCross[i])))
    for i in range(offsprings.size // 2):
        if offspringsOne == None and offspringsTwo == None:
            offspringsOne = np.array([offsprings[i]])
            offspringsTwo = np.array([offsprings[i+1]])
        elif i % 2 == 1:
            offspringsOne = np.concatenate((offspringsOne, np.array([offsprings[i+1]])))
            offspringsTwo = np.concatenate((offspringsTwo, np.array([offsprings[i+2]])))
        else:
            offspringsOne = np.concatenate((offspringsOne, np.array([offsprings[i]])))
            offspringsTwo = np.concatenate((offspringsTwo, np.array([offsprings[i+1]])))
    #set data for table Crossover.csv
    dataTwo  = np.array([indexOne, fenotypeOne, indexTwo, fenotypeTwo, pointOfCross, offspringsOne, offspringsTwo])
    fenotype = np.array([int(x, 2) for x in fenotypeOne])
    fenotype = np.concatenate((fenotype, np.array([int(x, 2) for x in fenotypeTwo])))
    draw_graph(rangex=[-2, 4], rangey=[-20, 30], name="Crossover", title="f(x) = 8x - 6x^2 - 5x^3 + 2x^4", points_x=fenotype)
    dataTwo = dataTwo.transpose()
    names = [u"Номер 1 батька",
             u"Хромосома 1",
             u"Номер 2 батька",
             u"Хромосома 2",
             u"Точка схрещування",
             u"Нащадок 1",
             u"Нащадок 2"
             ]
    table = pd.DataFrame(dataTwo, columns=names)
    table.to_csv("Crossover.csv",index=False)
    tableThree(dataTwo)

#calculate percent
def calPerc(i, n):
    return round(float(i) / float(n), 1)

#table with mutation
def tableThree(dataIn):
    newPopulation = dataIn[:, 5]
    newPopulation = np.concatenate((newPopulation, dataIn[:, 6]))

    z = 0
    n = newPopulation.size * k
    mutatedGen = np.array([False]*newPopulation.size)
    percent = calPerc(z, n)
    mutatedPopultation = np.array(newPopulation)
    while percent < pm and calPerc(z+1, n) < pm:
        for i in range(newPopulation.size):
            yes = random.random()
            if yes > 0.5:
                if mutatedGen[i] == False:
                    mutatedGen[i] = True
                    z+=1
                    x = random.randint(0, len(newPopulation[i]) - 1)
                    gen = newPopulation[i][x]
                    str1 = newPopulation[i][:x]
                    str2 = newPopulation[i][x+1:]
                    if gen == "1":
                        gen = "0"
                    else :
                        gen = "1"
                    s = str1 + gen + str2
                    mutatedPopultation[i] = s
                    percent = calPerc(z, n)
                    if percent >= pm or calPerc(z+1, n) >= pm:
                        break
    #set data for table mutation.csv
    index = np.array([str(x) + "'" for i in range(1, N+1)])
    data = np.array([index, newPopulation, mutatedPopultation])
    fenotype = np.array([int(x, 2) for x in mutatedPopultation])
    draw_graph(rangex=[-2, 4], rangey=[-20, 30], name="Mutation", title="f(x) = 8x - 6x^2 - 5x^3 + 2x^4",
               points_x=fenotype)
    data = data.transpose()
    names = [u"Номер особини в популяції",
             u"Хромосома до мутації",
             u"Хромосома після мутації"
             ]
    table = pd.DataFrame(data, columns=names)
    table.to_csv("mutation.csv", index=False)
    tableFour(data)

#table with new generation after mutation
def tableFour(dataIn):
    index = np.array([str(x) + "''" for x in range(1, N+1)])
    newPopulation = dataIn[:, 2]
    fenotype = np.array([int(x, 2) for x in newPopulation])
    koef = np.array([f(x, -1) for x in fenotype]) #send -1 to function because we find minimum
    data = np.array([index, newPopulation, fenotype, koef])
    data = data.transpose()
    draw_graph(rangex=[-2, 4], rangey=[-20, 30], name="new_population", title="f(x) = 8x - 6x^2 - 5x^3 + 2x^4",
               points_x=fenotype)
    names = [u"Номер особини в популяції",
             u"Хромосоми",
             u"Фенотип",
             u"Коефіцієнт пристосованості"
             ]
    table = pd.DataFrame(data, columns=names)
    table.to_csv("NewGeneration.csv", index=False)

    file = open("New_generation.txt", "w")
    sumKoef = sum(koef)
    values[1] = sumKoef / float(N)
    file.write(u"Середнє значення коефіцієнтів пристосованості: {}\n".format(values[1]))
    file.write(u"Максимальне значення коефіцієнта пристосованості: {}".format(values[2]))
    file.close()

#analise our results
def analise():
    file = open("result.txt", "w")
    if values[0] < values[1]:
        file.write(u"З результатів видно, що популяція нащадків має більше середнє значення коефіцієнта пристосованості.\nЗ цього випливає, що популяція нащадків є кращою.")
    elif values[0] > values[1]:
        file.write(u"З результатів видно, що популяція нащадків має менше середнє значення коефіцієнта пристосованості.\nЗ цього випливає, що популяція нащадків є гіршою.")
    else:
        file.write(u"З результатів видно, що популяція нащадків не змінилась.")
    file.close()

if __name__ == "__main__":
    draw_graph(rangex=[-2, 4], rangey=[-20, 30], name="graph", title="f(x) = 8x - 6x^2 - 5x^3 + 2x^4")
    tableOne()
    analise()