
import pandas as pd
import os
import requests
import csv
import numpy as np
import matplotlib.pyplot as plt
import pypfopt as pyp
import seaborn as sns
import datetime
import yfinance as yf
idx = pd.IndexSlice
from finam import Exporter, Market, LookupComparator,Timeframe
import pandas_datareader.data as web
import pyfolio as pf
import backtrader as bt
import requests
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#set period
start_date = '2014-10-30'
end_date = '2021-09-05'
fromdate=datetime.datetime(2014, 10, 30)
todate=datetime.datetime(2021, 9, 5)


# get connection to MOEX database 
URL='https://www.moex.com/a7798'
mcap_df = pd.read_html(URL,skiprows=0, index_col=None)
quant=(mcap_df[0].shape[0])
# get table Market Capitalization Рыночная капитализация
mcap = pd.DataFrame(
    index=range(quant),
    columns=['ticker', 'name', 'type', 'market_cap']
)
z=0
for i in range(quant):
    mcap['ticker'][z] = mcap_df[0].values[z][0]
    mcap['name'][z] = mcap_df[0].values[z][1]
    mcap['type'][z] = mcap_df[0].values[z][2]
    mcap['market_cap'][z] = float(str(mcap_df[0].values[z][6]).replace(' ','').replace(',','.'))
    z+=1
mcap=mcap.sort_values(by='market_cap', ascending=False)
#mcap.to_csv('mcap_ru.csv')
mcap=pd.read_csv('mcap_ru.csv', delimiter=',', index_col=[0])
print(mcap)
# get connection to MOEX database Рентабельность
URL='https://www.moex.com/ru/listing/profitability-xls.aspx'
headers={'User-Agent': 'Mozilla/5.0'}
r = requests.get(URL, headers=headers, verify=False)
profitability_df = pd.read_excel(URL,
                    #   sheet_name='profitabilty',
                       skiprows=0,
                       skipfooter=0
)
print(profitability_df.head(2))

# get connection to MOEX database Рентабельность
#URL='https://www.moex.com/ru/listing/dividend-yield.aspx'
#dividends_df = pd.read_excel(URL,
#                    #   sheet_name='profitabilty',
#                       skiprows=0,
#                       skipfooter=0
#)
#print(dividends_df)

#create tickers list
tickers = [ticker[:] for ticker in mcap.ticker]
mcap_dict = {ticker[:] : cap for ticker, cap in zip(mcap['ticker'].values, mcap['market_cap'].values)}


#get index history
ticker = 'RTS'
exporter = Exporter()
asset = exporter.lookup(name=ticker, market=Market.FUTURES)
asset_id = asset[asset['name'] == ticker].index[0]
data = exporter.download(asset_id, market=Market.FUTURES, timeframe=Timeframe.DAILY)
data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
rts = pd.DataFrame(data)
rts = rts.set_index('<DATE>')
rts=rts['<CLOSE>']
#x=rts.rename({'adj_close':'na'}, axis=1)
rts.index.names = ['Date']
rts= rts[start_date:end_date]
rts=rts.dropna()
print(rts.head(2))
print(rts)
ticker = 'IMOEX.ME'
rts = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
#rts1=pd.DataFrame(btc_prices)
print(rts.head(2))


#get history data from finam
exporter = Exporter()
prices=pd.DataFrame(
    index=pd.date_range(start_date,end_date,freq='D')
)
prices=prices[prices.index.isin(rts.index)]
result=pd.DataFrame(
    columns=['ticker','Date','High','Low','Open','Close','Volume']
)
result=result.set_index(['ticker', 'Date'])
tickers_us= ['LGIH','JOUT','LPX','MBUU','UFPI','PKI','SNX','NVR','PATK','JBL','FIX',
'LCII','CE','MUSA','HCA','KFRC','PLUS','COO','OTTR','GTY','LH']
z=0
for i in tickers:
    if z<=25 and tickers[z] !='BTC-USD' and tickers[z] !='ETH-USD'  and tickers[z] !='LTC-USD' and tickers[z] !='XMR-USD':
        ticker=tickers[z]
        asset = exporter.lookup(code=ticker, market=Market.SHARES)
        asset_id = asset[asset['code'] == ticker].index[0]
        try:
            data = exporter.download(asset_id, market=Market.SHARES, timeframe=Timeframe.DAILY)
            data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
            asset_df = data.set_index('<DATE>')
            asset_df.index.names = ['Date']
            candles= asset_df[start_date:end_date]
            asset_df[ticker]=asset_df['<CLOSE>']
            asset_df= asset_df[start_date:end_date].loc[:,ticker].sort_index()
            if len(asset_df)>0.9*len(rts) and asset_df.values[0] !=0 and asset_df.values[-1] !=0 and (asset_df.values[-1]/asset_df.values[0]) > 2.2:
                prices = pd.concat([prices, asset_df], axis=1)
                candles=candles[candles.index.isin(rts.index)]
                candles = candles.reset_index()
                candles= candles.loc[:,['Date', '<OPEN>','<HIGH>', '<LOW>', '<CLOSE>','<VOL>']]
                candles['ticker']=ticker
                print(ticker)
                candles=candles.set_index(['ticker', 'Date'])
                candles.columns=['Open','High','Low','Close','Volume']
                result=result.append(candles)
        except:
            print('data not found...')
    z+=1

prices_sel=pd.read_csv('prices_sel.csv', index_col=[0], parse_dates=True)
result_sel=pd.read_csv('result_sel.csv', index_col=[0,1], parse_dates=True)
prices = pd.concat([prices, prices_sel], axis=1)
result=result.append(result_sel)
prices_sel=pd.read_csv('prices111.csv', index_col=[0], parse_dates=True)
result_sel=pd.read_csv('result111.csv', index_col=[0,1], parse_dates=True)
prices = pd.concat([prices, prices_sel], axis=1)
result=result.append(result_sel)
print(prices.head(2))
print(result.head(2))
rts=rts[rts.index.isin(prices.index)]
prices=prices[prices.index.isin(rts.index)]
prices.to_csv('prices.csv')
result.to_csv('result.csv')
selected_tickers=prices.columns.tolist()
print(selected_tickers)

#create selected tickers list and dictionary
tickers = [ticker[:] for ticker in selected_tickers]
mcap1=mcap['ticker']
mcap_selected=mcap.loc[mcap['ticker'].isin(tickers)]
print(mcap_selected)
mcap_dict = {ticker[:] : cap for ticker, cap in zip(mcap_selected['ticker'].values, mcap_selected['market_cap'].values)}
print(mcap_dict)

# calculate asset covariance and delta
# market-implied risk premium, which is the market’s excess return divided by its variance
S = pyp.risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = pyp.black_litterman.market_implied_risk_aversion(rts, risk_free_rate=0.05796)
print(delta)

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(prices.pct_change().corr(method ='spearman'), ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
ax.set_title('Assets Correlation Matrix')
plt.savefig('chart1', dpi=300)

# calculate prior - market implied retunrs
market_prior = pyp.black_litterman.market_implied_prior_returns(mcap_dict, delta, S)
market_prior.name = 'Prior'
print(market_prior)

# plot prior
market_prior.plot.barh(figsize=(12,6), title = 'Priors - Market Implied Returns',grid=True);
plt.savefig('chart2', dpi=300)

# provide absolute views from consensus 
#URL ='https://bcs-express.ru/targets'
#URL2 ='https://quote.rbc.ru/catalog/?type=share&sort=forecast'
#headers={'User-Agent': 'Mozilla/5.0'}
#browser = webdriver.Chrome()
#browser.get(site)


#get views from file
view_confidence = pd.read_csv('views_all.csv', delimiter=';', index_col=[0])
view_confidence = pd.DataFrame(view_confidence, columns=[ 'View', 'Confidences'])
view_confidence=view_confidence.loc[view_confidence.index.isin(selected_tickers)]
print(view_confidence.head(25))

views_dict = {ind : view_confidence['View'][ind] for ind in view_confidence.index}
print(views_dict)

# run the Bl model
bl = pyp.BlackLittermanModel(S, pi=market_prior, absolute_views=views_dict)

bl_return = bl.bl_returns()
bl_return.name = 'Posterior'

mu = pyp.expected_returns.mean_historical_return(prices)
mu.name = 'Historical'

returns_df = pd.DataFrame([market_prior, mu, bl_return, pd.Series(views_dict)], 
             index=['Prior', 'Historical','Posterior', 'Views']).T
print(returns_df)

returns_df.to_csv('returns.csv', header=True,)
returns_df = pd.read_csv('returns.csv', index_col=[0], )

returns_df.plot.bar(figsize=(14,6), title = 'Returns Estimates - Prior, Historical, Posterior, Views', grid=True);
plt.savefig('chart3', dpi=300)

S_bl = bl.bl_cov()
S_bl.to_csv('S_bl.csv')
S_bl = pd.read_csv('S_bl.csv',  index_col=[0])

confidences = list(view_confidence.Confidences)
print(confidences)

bl_confi = pyp.BlackLittermanModel(S, pi=market_prior, 
                                   absolute_views=views_dict, 
                                   omega="idzorek", view_confidences=confidences)


bl_return_confi = bl_confi.bl_returns()
bl_return_confi.name = 'Posterior_confidence'

returns_df = pd.DataFrame([market_prior, mu, bl_return, pd.Series(views_dict), bl_return_confi], 
             index=['Prior', 'Historical','Posterior', 'Views', 'Posterior_confidence']).T
print(returns_df)

returns_df.to_csv('returns.csv', header=True,)
returns_df = pd.read_csv('returns.csv', index_col=[0],)

returns_df.plot.bar(figsize=(14,6), 
                    title = 'Returns Estimates - Prior, Historical, Posterior, Views, Posterior-confidence', grid=True);
plt.savefig('chart4', dpi=300)

S_bl_confi = bl_confi.bl_cov()
S_bl_confi.to_csv('S_bl_confi.csv')
S_bl_confi = pd.read_csv('S_bl_confi.csv',  index_col=[0])
S_bl_confi

ef = pyp.EfficientFrontier(bl_return_confi, S_bl_confi, weight_bounds=(0, 0.1))
ef.add_objective(pyp.objective_functions.L2_reg, gamma=0.1)
weights = ef.min_volatility()
ef.portfolio_performance(verbose=True), print('\n')
wt_min_vola = pd.DataFrame([weights],columns=weights.keys()).T * 100

# write it to csv for part 2
wt_min_vola.to_csv('wt_min_vola_wts.csv')
wt_min_vola = pd.read_csv('wt_min_vola_wts.csv',  index_col=[0])

print ('Weights in Percentage ********************')
print(wt_min_vola.round(4))

wt_min_vola.plot.bar(figsize=(14,6), 
                    title = 'Asset Allocation Based on BL with Confidence Matrix', grid=True,legend=False);
plt.ylabel('Percentage')
plt.savefig('chart5', dpi=300)

wt_min_vola = pd.read_csv('wt_min_vola_wts.csv',  index_col=[0])
print(wt_min_vola)

wt_min_vola.plot.bar(figsize=(14,6), 
                    title = 'Asset Allocation Based on BL with Confidence Matrix', grid=True,legend=False);
plt.ylabel('Percentage')
plt.savefig('chart6', dpi=300)


# get data for the backtesting

prices=result

assets_param = [(ind, wt_min_vola.loc[ind][0]) for ind in wt_min_vola.index]
print(assets_param)

class Strategy(bt.Strategy):
    # parameters for inputs    
    params = dict(
        assets = [],
        rebalance_months = [1,4,7,10]
    )
 
    # initialize
    def __init__(self):
        # create a dictionary of ticker:{'rebalanced': False, 'target_percent': target%}
        self.rebalance_dict = dict()
        for i, d in enumerate(self.datas):
            self.rebalance_dict[d] = dict()
            self.rebalance_dict[d]['rebalanced'] = False
            for asset in self.p.assets:
                if asset[0] == d._name:
                    self.rebalance_dict[d]['target_percent'] = asset[1]
 
    def next(self):
        # rebalance for the month in the list
        for i, d in enumerate(self.datas):
            dt = d.datetime.datetime()
            dname = d._name
            pos = self.getposition(d).size
 
            if dt.month in self.p.rebalance_months and self.rebalance_dict[d]['rebalanced'] == False:
                print('{} Sending Order: {} | Month {} | Rebalanced: {} | Pos: {}'.
                      format(dt, dname, dt.month,
                             self.rebalance_dict[d]['rebalanced'], pos ))
            
                self.order_target_percent(d, target=self.rebalance_dict[d]['target_percent']/100)
                self.rebalance_dict[d]['rebalanced'] = True
 
            # Reset the flage
            if dt.month not in self.p.rebalance_months:
                self.rebalance_dict[d]['rebalanced'] = False
                
    # notify the order if completed
    def notify_order(self, order):
        date = self.data.datetime.datetime().date()
 
        if order.status == order.Completed:
            print('{} >> Order Completed >> Stock: {},  Ref: {}, Size: {}, Price: {}'.
                  format(date, order.data._name, order.ref, order.size,
                         'NA' if not order.price else round(order.price,5)
                        ))

    # notify the trade if completed        
    def notify_trade(self, trade):
        date = self.data.datetime.datetime().date()
        if trade.isclosed:
            print('{} >> Notify Trade >> Stock: {}, Close Price: {}, Profit, Gross {}, Net {}'.
                  format(date, trade.data._name, trade.price, round(trade.pnl,2),round(trade.pnlcomm,2))
                 )


startcash = 500000

# 0.4% commission
commission = 0.004

#Create an instance of cerebro
cerebro = bt.Cerebro()

cerebro.broker.setcash(startcash)

# orders will not be checked to see if you can afford it before submitting them
cerebro.broker.set_checksubmit(False)

cerebro.broker.setcommission(commission=commission)

TICKERS = list(prices.index.get_level_values('ticker').unique())
print(TICKERS)

for ticker, data in prices.groupby(level=0):
    if ticker in TICKERS:
        print(f"Adding ticker: {ticker}")
        data = bt.feeds.PandasData(dataname=data.droplevel(level=0),
                                   name=str(ticker),
                                   fromdate=fromdate,
                                   todate=todate,
                                   plot=False)
        cerebro.adddata(data)

cerebro.addstrategy(Strategy, assets=assets_param)

# add analyzers
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
# Run the strategy. 
results = cerebro.run(stdstats=True, tradehistory=False)
# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

strat = results[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')

returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
returns.name = 'Strategy'
print(returns.head(2))

benchmark = 'IMOEX.ME' # S&P BSE Sensex
benchmark_rets= web.DataReader(benchmark, 'yahoo', start=start_date,)['Adj Close'].pct_change().dropna()
benchmark_rets.index = benchmark_rets.index.tz_localize('UTC') 
benchmark_rets = benchmark_rets.filter(returns.index)
benchmark_rets.name = 'MICEX index'
benchmark_rets.head(2)

benchmark = 'IMOEX.ME' # S&P BSE Sensex
benchmark_prices = web.DataReader(benchmark, 'yahoo', start=start_date,)['Adj Close']
benchmark_prices = benchmark_prices.asfreq('D', method='ffill')
benchmark_prices.index = benchmark_prices.index.tz_localize('UTC')
benchmark_prices = benchmark_prices.filter(returns.index)
benchmark_prices.head(5)

benchmark_prices = (benchmark_prices/benchmark_prices.iloc[0]) * startcash
benchmark_prices.head()

benchmark1 = '^GSPC' # S&P BSE Sensex
benchmark_rets1= web.DataReader(benchmark1, 'yahoo', start=start_date,)['Adj Close'].pct_change().dropna()
benchmark_rets1.index = benchmark_rets1.index.tz_localize('UTC')
benchmark_rets1 = benchmark_rets1.filter(returns.index)
benchmark_rets1.name = '^GSPC'
benchmark_rets1.head(2)


benchmark1 = '^GSPC' # S&P BSE Sensex
benchmark_prices1 = web.DataReader(benchmark1, 'yahoo', start=start_date,)['Adj Close']
benchmark_prices1 = benchmark_prices1.asfreq('D', method='ffill')
benchmark_prices1.index = benchmark_prices1.index.tz_localize('UTC')
benchmark_prices1 = benchmark_prices1.filter(returns.index)
benchmark_prices1.head(5)

benchmark_prices1 = (benchmark_prices1/benchmark_prices1.iloc[0]) * startcash
benchmark_prices1.head()

portfolio_value = returns.cumsum().apply(np.exp) * startcash

# Visulize the output
fig, ax = plt.subplots(2, 1, sharex=True, figsize=[14, 8])

# portfolio value
portfolio_value.plot(ax=ax[0], label='Strategy')
benchmark_prices.plot(ax=ax[0], label='Benchmark - MICEX index')
benchmark_prices1.plot(ax=ax[0], label='Benchmark - ^GSPC index')
ax[0].set_ylabel('Portfolio Value')
ax[0].grid(True)
ax[0].legend()

# daily returns
returns.plot(ax=ax[1], label='Strategy', alpha=0.5)
benchmark_rets.plot(ax=ax[1], label='Benchmark - MICEX index', alpha=0.5)
benchmark_rets1.plot(ax=ax[1], label='Benchmark - ^GSPC index', alpha=0.5)
ax[1].set_ylabel('Daily Returns')

fig.suptitle('Black–Litterman Portfolio Allocation vs S&P MICEX vs ^GSPC index', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
fig.savefig('chart9', dpi=300)


# get performance statistics for strategy
pf.show_perf_stats(returns,)

# get performance statistics for benchmark
pf.show_perf_stats(benchmark_rets)

# plot performance for strategy
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8),constrained_layout=True)
axes = ax.flatten()

#pf.plot_drawdown_periods(returns=returns, ax=axes[0])
axes[0].grid(True)
pf.plot_rolling_returns(returns=returns,
                        factor_returns=benchmark_rets,
                        ax=axes[1], title='Strategy vs MICEX')
axes[1].grid(True)
pf.plot_drawdown_underwater(returns=returns, ax=axes[2])
axes[2].grid(True)
pf.plot_rolling_sharpe(returns=returns, ax=axes[3])
axes[3].grid(True)
fig.suptitle('BL Portfolio vs MICEX', fontsize=12, y=0.990)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('chart7', dpi=300)

# plot performance
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9),constrained_layout=True)
axes = ax.flatten()

pf.plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[0])
axes[0].grid(True)

pf.plot_rolling_volatility(returns=returns, factor_returns=benchmark_rets,ax=axes[1])
axes[1].grid(True)

pf.plot_annual_returns(returns=returns, ax=axes[2])
axes[2].grid(True)

pf.plot_monthly_returns_heatmap(returns=returns, ax=axes[3],)
fig.suptitle('BL Portfolio vs MICEX', fontsize=16, y=1.0)


plt.tight_layout()
plt.savefig('chart8', dpi=300)


