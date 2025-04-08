# 📊 트레이딩 성과 보고서

생성 시간: {{ timestamp }}

## 📈 종합 성과 지표

- 총 수익률: {{ "%.2f"|format(metrics.total_return * 100) }}%
- 승률: {{ "%.2f"|format(metrics.win_rate * 100) }}%
- 수익 팩터: {{ "%.2f"|format(metrics.profit_factor) }}
- 최대 낙폭: {{ "%.2f"|format(metrics.max_drawdown * 100) }}%
- 샤프 비율: {{ "%.2f"|format(metrics.sharpe_ratio) }}
- 소르티노 비율: {{ "%.2f"|format(metrics.sortino_ratio) }}
- 칼마 비율: {{ "%.2f"|format(metrics.calmar_ratio) }}

## 📊 거래 통계

- 총 거래 수: {{ metrics.total_trades }}
- 수익 거래: {{ metrics.winning_trades }}
- 손실 거래: {{ metrics.losing_trades }}
- 평균 수익률: {{ "%.2f"|format(metrics.average_return * 100) }}%
- 수익률 표준편차: {{ "%.2f"|format(metrics.return_std * 100) }}%

## 💰 수익/손실 분석

- 평균 수익: {{ "%.2f"|format(metrics.average_win) }}
- 평균 손실: {{ "%.2f"|format(metrics.average_loss) }}
- 최대 수익: {{ "%.2f"|format(metrics.largest_win) }}
- 최대 손실: {{ "%.2f"|format(metrics.largest_loss) }}
- 평균 보유 시간: {{ metrics.average_hold_time }}

## 📈 포지션별 성과

{% for symbol, metrics in position_metrics.items() %}
### {{ symbol }}

- 총 수익률: {{ "%.2f"|format(metrics.total_return * 100) }}%
- 승률: {{ "%.2f"|format(metrics.win_rate * 100) }}%
- 수익 팩터: {{ "%.2f"|format(metrics.profit_factor) }}
- 총 거래 수: {{ metrics.total_trades }}
- 평균 보유 시간: {{ metrics.average_hold_time }}
- 최대 낙폭: {{ "%.2f"|format(metrics.max_drawdown * 100) }}%

{% endfor %} 