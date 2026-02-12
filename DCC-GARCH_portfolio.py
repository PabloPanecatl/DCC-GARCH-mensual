import pandas as pd
import numpy as np
from ib_insync import *
from arch import arch_model
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ===================================================
# PARTE 1: MATEMÁTICA CUANTITATIVA (DCC-GARCH Proxy)
# ===================================================

def fit_garch_ewma_dcc(returns, lambda_corr=0.94):
    """
    Implementación robusta de DCC-GARCH para rebalanceo.
    1. Ajusta GARCH(1,1) Univariado a cada activo.
    2. Calcula residuos estandarizados.
    3. Usa EWMA en los residuos para estimar la Correlación Dinámica (DCC Proxy).
    4. Reconstruye la matriz de Covarianza condicional para T+1.
    """
    print("Ajustando GARCH(1,1) y Correlación Dinámica...")
    
    volatilities = []
    std_resids = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    # 1. GARCH Univariado
    latest_vols = [] # D_t para el último periodo
    for col in returns.columns:
        # Asumimos media cero para simplificar el GARCH en rebalanceo diario
        am = arch_model(returns[col], vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
        res = am.fit(disp='off')
        
        # Volatilidad condicional (Sigma_t)
        cond_vol = res.conditional_volatility
        volatilities.append(cond_vol)
        
        # Residuos Estandarizados (epsilon_t = r_t / sigma_t)
        std_resids[col] = returns[col] / cond_vol
        
        # Proyección de volatilidad a T+1 (h_t+1)
        # Forecast horizon=1
        forecast = res.forecast(horizon=1)
        next_vol = np.sqrt(forecast.variance.iloc[-1].values[0])
        latest_vols.append(next_vol)

    latest_vols = np.array(latest_vols) # Diagonal de D_t+1
    
    # 2. Correlación Dinámica (EWMA sobre Residuos Estandarizados)
    # Qt = (1-lambda) * (e_t-1 * e'_t-1) + lambda * Qt-1
    # Usamos pandas ewm para vectorizar esto eficientemente
    corr_dynamic = std_resids.ewm(alpha=(1 - lambda_corr), adjust=False).corr()
    
    # Extraemos la última matriz de correlación (T) para usarla en T+1
    # El objeto ewm cov/corr devuelve un MultiIndex, tomamos el último bloque fecha
    last_date = returns.index[-1]
    R_next = corr_dynamic.loc[last_date].values
    
    # 3. Reconstruir Covarianza: Sigma = D * R * D
    # D es una matriz diagonal con las volatilidades predichas
    D_next = np.diag(latest_vols)
    Cov_next = D_next @ R_next @ D_next
    
    return Cov_next

def get_expected_returns_js(returns, lookback=60):
    """
    Estimador James-Stein simplificado.
    Encoge los promedios individuales hacia la media global del portafolio.
    """
    # 1. Medias simples de la ventana
    mu_sample = returns.tail(lookback).mean()
    
    # 2. Gran Media (Grand Mean) de todos los activos
    mu_global = mu_sample.mean()
    
    # 3. Factor de Shrinkage (Heurística: 0.5 es un buen punto de partida conservador)
    # En modelos avanzados se calcula dinámicamente según la volatilidad.
    phi = 0.5 
    
    # 4. Cálculo
    mu_shrunk = (1 - phi) * mu_sample + phi * mu_global
    
    return mu_shrunk.values

def optimize_portfolio_sharpe(cov_matrix, expected_returns):
    """Optimización de Media-Varianza maximizando Sharpe Ratio."""
    n_assets = len(expected_returns)
    
    def objective(weights):
        port_ret = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret / port_vol) if port_vol > 0 else 0

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    init_guess = [1./n_assets] * n_assets
    
    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

# ==========================================
# PARTE 2: SACAR DATOS DE IBKR Y EJECUTAR
# ==========================================

def get_ibkr_data(ib, tickers, duration='2 Y', bar_size='1 day'):
    """Descarga datos históricos directamente desde los servidores de IB."""
    print(f"Solicitando historia a IBKR ({duration})...")
    close_prices = pd.DataFrame()
    
    for ticker in tickers:
        contract = Stock(ticker, 'SMART', 'USD')
        # Solicitud síncrona (bloqueante para asegurar datos antes de calcular)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            formatDate=1
        )
        if bars:
            df = util.df(bars)
            df.set_index('date', inplace=True)
            close_prices[ticker] = df['close']
        else:
            print(f"No se encontraron datos para {ticker}")
            
    return close_prices.dropna()

def get_market_price(ib, contract):
    """Obtiene snapshot del precio actual (Bid/Ask) para definir el límite."""
    # Solicitamos datos de mercado en vivo (Snapshot)
    ticker_data = ib.reqMktData(contract, '', False, False)
    ib.sleep(2) # Esperar a que lleguen los ticks
    
    # Lógica de precio: Intentar MidPrice, si falla, LastPrice, si falla, Close anterior.
    bid = ticker_data.bid
    ask = ticker_data.ask
    last = ticker_data.last
    close = ticker_data.close
    
    if (bid > 0) and (ask > 0):
        return bid, ask
    elif last > 0:
        return last, last # Spread asumed 0 si no hay bid/ask
    else:
        return close, close # Fallback final

# ==========================================
# PARTE 3: EJECUCIÓN PRINCIPAL
# ==========================================

def main():
    # CONFIGURACIÓN
    # El chiste es agarrar acciones de diferentes sectores para q el modelo tenga de dónde escoger
    TICKERS = ['F', 'PFE', 'HPE', 'BAC'] 
    PORT = 7496 
    
    ib = IB()
    try:
        print("Conectando a IBKR (Puerto 7496)...")
        # Usamos clientId=102 pq la regué con 101 antes
        ib.connect('127.0.0.1', PORT, clientId=102) 
        ib.reqMarketDataType(3) 
        print("Usando datos diferidos (15 min de retraso).")
    except Exception as e:
        print(f"Error conectando: {e}")
        return

    # 1. Obtener Valor de la Cuenta
    acct_vals = ib.accountSummary()
    # NetLiquidation = Efectivo + Valor de Acciones actuales
    net_liq = next((float(v.value) for v in acct_vals if v.tag == 'NetLiquidation'), 0.0)
    print(f"Capital Total (Equity): ${net_liq:,.2f}")

    # 2. Obtener Datos Históricos
    prices = get_ibkr_data(ib, TICKERS)
    if prices.empty:
        print("Error: No se pudieron descargar precios.")
        ib.disconnect()
        return

    # 3. Calcular Modelo (Math Layer)
    log_returns = np.log(prices).diff().dropna()
    cov_next = fit_garch_ewma_dcc(log_returns)
    mu_next = get_expected_returns_js(log_returns, lookback=60)
    
    # Optimización
    weights = optimize_portfolio_sharpe(cov_next, mu_next)
    target_weights = dict(zip(prices.columns, weights))
    
    # Limpieza de pesos (filtrar < 2% y renormalizar)
    target_weights = {k: v for k, v in target_weights.items() if v > 0.02}
    total_w = sum(target_weights.values())
    if total_w > 0:
        target_weights = {k: v/total_w for k, v in target_weights.items()}
    else:
        print("No hay pesos válidos.")
        ib.disconnect()
        return

    print("\nPesos Objetivo (DCC-GARCH):")
    for t, w in target_weights.items():
        print(f"   {t}: {w:.1%}")

    # 4. Ejecución (Execution Layer)
    current_positions = {p.contract.symbol: p.position for p in ib.positions()}
    
    print("\n⚙️ Generando Órdenes Limitadas...")
    
    # --- VARIABLES PARA EL REPORTE FINAL ---
    total_invested_real = 0.0
    # ---------------------------------------

    for ticker in TICKERS:
        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        current_qty = current_positions.get(ticker, 0)
        
        # Obtener precio robusto (Vivo o Histórico)
        bid, ask = get_market_price(ib, contract)
        if (bid > 0 and ask > 0):
            mid_price = (bid + ask) / 2
            data_source = "En vivo"
        else:
            mid_price = prices[ticker].iloc[-1]
            data_source = "Histórico"

        mid_price = float(mid_price) # Asegurar float

        if mid_price <= 0 or np.isnan(mid_price):
            print(f"   ❌ {ticker}: Precio inválido. Saltando.")
            continue
            
        # Cálculo de asignación
        target_val = net_liq * target_weights.get(ticker, 0.0)
        target_qty = int(target_val / mid_price) # Aquí ocurre el truncamiento (sobra cash)
        
        # ACUMULADOR: Sumamos el valor REAL de lo que tendremos en acciones
        total_invested_real += (target_qty * mid_price)
        
        diff_qty = target_qty - current_qty
        
        if diff_qty == 0:
            print(f"   ✅ {ticker}: Posición óptima (Hold {target_qty}).")
            continue
            
        action = 'BUY' if diff_qty > 0 else 'SELL'
        abs_qty = abs(diff_qty)
        
        # Slippage Control
        slippage_tol = 0.005 if data_source == "Histórico" else 0.001
        if action == 'BUY':
            lmt_price = round(mid_price * (1 + slippage_tol), 2)
        else:
            lmt_price = round(mid_price * (1 - slippage_tol), 2)

        order = LimitOrder(action, abs_qty, lmt_price)
        order.tif = 'GTC'
        order.transmit = False # Chécalo en TWS antes de enviar
        
        ib.placeOrder(contract, order)
        print(f"   🚀 {action} {abs_qty} {ticker} @ {lmt_price} ({data_source}: {mid_price:.2f})")

    # --- REPORTE FINANCIERO FINAL ---
    cash_left = net_liq - total_invested_real
    pct_invested = (total_invested_real / net_liq) * 100
    pct_cash = (cash_left / net_liq) * 100
    
    print("\n" + "="*40)
    print("      🧾 RESUMEN DE PORTAFOLIO")
    print("="*40)
    print(f"💵 Capital Total:      ${net_liq:,.2f}")
    print(f"🏗️  Capital Invertido:  ${total_invested_real:,.2f} ({pct_invested:.1f}%)")
    print(f"🐖 Cash 'Suelto':      ${cash_left:,.2f} ({pct_cash:.1f}%)")
    print("="*40 + "\n")
    
    print("Ahí nomás quedó")
    ib.disconnect()

if __name__ == "__main__":
    main()
