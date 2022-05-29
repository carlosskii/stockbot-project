import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import numpy as np
import random
import math
from dataclasses import dataclass, field
import yfinance as yf
from datetime import date, timedelta
import os.path
import threading
import time
import copy
import dill

# INTERVAL: Time in milliseconds between each render cycle.
INTERVAL = 100

# MUTATION_VARIATION: Whevener a mutation occurs, this is the maximum
# amount of variation that can be applied to the weight and bias.
# Ex. If this is 0.1, then the amount added to weight and bias will be in
# the range [-0.1, 0.1].
MUTATION_VARIATION = 1

# BOT_COUNT_SQRT: The square root of the amount of bots in the simulation.
# Ex. If this is 10, there will be 100 bots in the simulation for each generation.
BOT_COUNT_SQRT = 10

# BOTTOM_GRAPH_QUANTITY: The amount of generations shown in the bottom scatterplot.
# Ex. IF this is 25, the bottom scatterplot will show the last 25 generations. When
# more simulations are ran, the oldest generations will be removed from the scatterplot.
BOTTOM_GRAPH_QUANTITY = 25

# RESET_BOTS: If this is true, the bots will be reset when the program starts, regardless
# of bot backup files. If there are no backup files, the bots will be fully reset anyway.
RESET_BOTS = False

# STARTING_MONEY: The amount of money each bot starts with on each generation.
STARTING_MONEY = 1000

################################
# Misc
################################

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def clamp(a, b, c):
    a[a<b] = b
    a[a>c] = c
    return a

def on_close():
    plt.close()
    exit()

################################
# StockData
################################

@dataclass(order=True, eq=True, repr=True)
class StockData:
    ticker: str = "GOOGL"
    adjcloses: dict = field(default_factory=list, compare=True)
    start: str = "2017-01-01"
    end: str = "2017-01-02"
    interval: str = "1d"
    n = 0

    def __post_init__(self):
        if self.adjcloses == []:
            self.adjcloses = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval)["Adj Close"]
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n >= len(self.adjcloses):
            raise StopIteration
        result = self.adjcloses[self.n]
        self.n += 1
        return result
    
    def __len__(self):
        return len(self.adjcloses)

    def __getitem__(self, n):
        return self.adjcloses[n]

################################
# CreatureManager
################################

class CreatureManager:
    def __init__(self):
        self.creatures = [Creature() for i in range(100)]
        self.simdata = None
        self.final_results = []
        self.final_results_indexes = []
        self.n = 4
        self.j = 0
    
    def process_frame(self):
        if self.n == 4:
            self.stock_prices = random.choice(self.stock_price_options)
            self.simdata = np.zeros((min([len(i) for i in self.stock_price_options])-5, BOT_COUNT_SQRT ** 2))
        stock_prices = self.stock_prices
        for j in range(BOT_COUNT_SQRT ** 2):
            kn1 = (stock_prices[self.n] - stock_prices[self.n+1]) / stock_prices[self.n]
            k0 = (stock_prices[self.n-4] - stock_prices[self.n-3]) / stock_prices[self.n-4]
            k1 = (stock_prices[self.n-3] - stock_prices[self.n-2]) / stock_prices[self.n-3]
            k2 = (stock_prices[self.n-2] - stock_prices[self.n-1]) / stock_prices[self.n-2]
            k3 = (stock_prices[self.n-1] - stock_prices[self.n]) / stock_prices[self.n-1]
            k4 = self.n/len(stock_prices)
            self.creatures[j].decide_on_stock([kn1, k0, k1, k2, k3, k4], "AAPL", stock_prices[self.n])
            self.simdata[self.n-4][j] = self.creatures[j].money
    
    def plot_data(self):
        self.process_frame()

    def run_simulation(self, stock_prices):

        for j in range(1000000000000000000000000000):
            self.j = j

            for i in range(4, min([len(i) for i in stock_prices])-1):
                self.n = i
                self.plot_data()
                time.sleep(INTERVAL / 2000)
            self.n += 1
            self.final_results.extend(self.simdata[-1][9::10])
            self.final_results_indexes.extend([j] * 10)
            self.final_results = self.final_results[-BOTTOM_GRAPH_QUANTITY*10:] if len(self.final_results) > BOTTOM_GRAPH_QUANTITY*10 else self.final_results
            self.final_results_indexes = self.final_results_indexes[-BOTTOM_GRAPH_QUANTITY*10:] if len(self.final_results_indexes) > BOTTOM_GRAPH_QUANTITY*10 else self.final_results_indexes
            newcreatures = []
            endmoney = [i.money for i in self.creatures]
            endstock = [i.stock_sold for i in self.creatures]
            endgoodsales = [i.good_sales for i in self.creatures]
            endmoney = np.lexsort((endstock, endgoodsales, endmoney))
            highcreatures = [self.creatures[i] for i in endmoney[-BOT_COUNT_SQRT:]]
            newcreatures.extend([i.breed(j, True) for i in highcreatures for j in highcreatures])
            self.creatures = newcreatures
            filenames = ["YES.bots", "YES2.bots"]
            with open(filenames[j%2], "wb") as wb:
                dill.dump([j, self.creatures], wb)
            with open("FINALDATA.bots", "wb") as wb:
                dill.dump(self.final_results, wb)

################################
# Creature
################################

class Creature:
    def __init__(self):
        # Change this to change the brain dimensions. Do not change the input and output size
        # unless you understand what they do.
        self.brain_design = [
            {"input_dim": 7, "output_dim": 128, "activation": "relu"},
            {"input_dim": 128, "output_dim": 96, "activation": "sigmoid"},
            {"input_dim": 96, "output_dim": 64, "activation": "relu"},
            {"input_dim": 64, "output_dim": 3, "activation": "sigmoid"},
        ]
        self.params_values = {}

        for idx, each in enumerate(self.brain_design):
            inSize = each["input_dim"]
            outSize = each["output_dim"]
            self.params_values[f"W{idx}"] = np.random.randn(inSize, outSize) * 0.1
            self.params_values[f"b{idx}"] = np.random.randn(outSize) * 0.1

        self.stock = {}
        self.stock_sold = 0
        self.money = STARTING_MONEY
        self.id = random.random()
        self.good_sales = 0
    
    def propogate_layer(self, WCurr, APrev, bCurr, activation="relu"):
        ZCurr = np.dot(APrev, WCurr) + bCurr

        if activation == "relu":
            afunc = relu
        else:
            afunc = sigmoid
        
        return afunc(ZCurr)

    def propogate(self, stock_values):
        ACurr = stock_values

        for idx, layer in enumerate(self.brain_design):
            APrev = ACurr

            afunc_curr = layer["activation"]
            WCurr = self.params_values[f"W{idx}"]
            bCurr = self.params_values[f"b{idx}"]
            ACurr = self.propogate_layer(WCurr, APrev, bCurr, afunc_curr)

        return list(ACurr)
    
    def add_mutation(self, mutchance):
        key = random.random()
        if key <= mutchance:
            for idx, _ in enumerate(self.brain_design):
                layer_size = self.params_values[f"W{idx}"].shape
                self.params_values[f"W{idx}"] += clamp(np.random.randn(layer_size[0], layer_size[1]), -MUTATION_VARIATION, MUTATION_VARIATION)
                self.params_values[f"b{idx}"] += clamp(np.random.randn(layer_size[1]), -MUTATION_VARIATION, MUTATION_VARIATION)
        return self
    
    def decide_on_stock(self, stock_values, ticker, stock_price):
        if self.money == STARTING_MONEY:
            quant_to_buy = 1
            if not ticker in self.stock:
                self.stock[ticker] = []
            for _ in range(quant_to_buy):
                self.stock[ticker].append(stock_price)
            self.money -= quant_to_buy * stock_price
        six = 0 if ticker in self.stock else 1
        if ticker in self.stock:
            for each in self.stock[ticker]:
                if each < stock_price:
                    six += 1
            six /= len(self.stock[ticker]) if len(self.stock[ticker]) != 0 else 1
        stock_values.append(six)
        tail = self.propogate(stock_values)
        self.evaluate(tail, ticker, stock_price)
    
    def evaluate(self, tail, ticker, stock_price):
        if ticker in self.stock and len(self.stock[ticker]) > 0:
            quant_to_sell = math.floor(len(self.stock[ticker]) * tail[1])
            made_good_sale = False
            for _ in range(quant_to_sell):
                if min(self.stock[ticker]) < stock_price:
                    made_good_sale = True
                self.stock[ticker].remove(min(self.stock[ticker]))
            self.money += stock_price * quant_to_sell * 0.9
            self.stock_sold += 1
            self.good_sales += 1 if made_good_sale else 0
        if self.money >= stock_price:
            quant_to_buy = math.floor((self.money / stock_price) * tail[2])
            if not ticker in self.stock:
                self.stock[ticker] = []
            for _ in range(quant_to_buy):
                self.stock[ticker].append(stock_price)
            self.money -= quant_to_buy * stock_price
    
    def breed(self, other, canMutate):
        newcreature = Creature()
        # If bot did absolutely nothing, toss it for a new one.
        if self.money == STARTING_MONEY:
            return newcreature
        for idx, _ in enumerate(self.brain_design):
            lsize = self.params_values[f"W{idx}"].shape
            maskW = np.random.random(lsize)
            maskB = np.random.random((lsize[1]))
            newcreature.params_values[f"W{idx}"] = np.where(maskW <= 0.8, self.params_values[f"W{idx}"], other.params_values[f"W{idx}"])
            newcreature.params_values[f"b{idx}"] = np.where(maskB <= 0.8, self.params_values[f"b{idx}"], other.params_values[f"b{idx}"])
        if canMutate:
            newcreature.add_mutation(0.2 if not (self.money > 995 and self.money < 1005) else 1)
        return newcreature
    
    def __neq__(self, other):
        return self.id != other.id


################################
# Run
################################

OFFSET = timedelta(weeks=12)

# Blue, Green, Yellow, Orange, Red, Pink, White
lscmcolors = [
    (0, 0, 1),
    (0, 0.5, 0),
    (1, 1, 0.3),
    (1, 0.5, 0),
    (1, 0, 0),
    (1, 0.7, 0.7),
    (1, 1, 1)
]

lscmm = LinearSegmentedColormap.from_list("mmmcolors", lscmcolors)

cmanage = CreatureManager()

if not RESET_BOTS and os.path.exists("YES.bots"):

    a = dill.load(open("YES.bots", "rb"))
    b = dill.load(open("YES2.bots", "rb"))

    cmanage.creatures = a[1] if a[0] > b[0] else b[1]

fig, (ax, ax2) = plt.subplots(2, 1)

fig.canvas.mpl_connect('close_event', on_close)

stockDataSamples = []
tickers = ["AAPL", "GOOGL"]

if os.path.exists("DATA.bots"):
    stockDataSamples = dill.load(open("DATA.bots", "rb"))
else:
    for ticker in tickers:
        for i in range(1, 35):
            START = date.today() - OFFSET*i
            END   = date.today() - OFFSET*(i-1)
            stockDataSamples.append(StockData(ticker=ticker, start=START.isoformat(), end=END.isoformat(), interval="1d"))
    dill.dump(stockDataSamples, open("DATA.bots", "wb"))

cmanage.stock_price_options = stockDataSamples
cmap = plt.cm.get_cmap("RdYlBu_r")

################################
# Render Section
################################

old_j = 0

def render():
    global old_j
    ax.clear()
    if cmanage.simdata is None: return
    n, bins, patches = ax.hist(cmanage.simdata[cmanage.n-5], 25)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bin_centers /= STARTING_MONEY*2
    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", lscmm(c))
    if cmanage.j != old_j:
        if len(cmanage.final_results) > 0:
            ax2.clear()
            xval = np.array(cmanage.final_results_indexes)
            yval = np.array(cmanage.final_results)
            ax2.scatter(xval, yval, c=yval, cmap=lscmm, norm=Normalize(0, STARTING_MONEY*2))
            old_j = copy.deepcopy(cmanage.j)
            ax2.set_ylim(0, STARTING_MONEY*2)
            ax2.hlines(STARTING_MONEY, min(xval)-1, min(xval) + BOTTOM_GRAPH_QUANTITY + 1)
            ax2.set_xlim(min(xval)-1, min(xval) + BOTTOM_GRAPH_QUANTITY + 1)
    plt.draw()

timer = fig.canvas.new_timer(interval=INTERVAL)
timer.add_callback(render)
timer.start()

ax.set_facecolor("black")
ax2.set_facecolor("black")
fig.colorbar(cm.ScalarMappable(norm=Normalize(0, STARTING_MONEY*2), cmap=lscmm), ax=[ax, ax2])

x = threading.Thread(target=cmanage.run_simulation, daemon=True, args=(stockDataSamples,))
x.start()

plt.show()