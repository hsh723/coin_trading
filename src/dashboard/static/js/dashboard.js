// 대시보드 JavaScript 코드

// 전역 변수
let socket = null;
let updateInterval = null;

// DOM 로드 완료 후 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 웹소켓 연결
    socket = io();
    
    // 사이드바 토글
    document.getElementById('toggleSidebar').addEventListener('click', toggleSidebar);
    
    // 네비게이션 링크 이벤트
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = e.target.closest('.nav-link').dataset.section;
            showSection(section);
        });
    });
    
    // 폼 제출 이벤트
    document.getElementById('strategyForm').addEventListener('submit', handleStrategySubmit);
    document.getElementById('notificationForm').addEventListener('submit', handleNotificationSubmit);
    
    // 초기 데이터 로드
    loadInitialData();
    
    // 주기적 업데이트 시작
    startPeriodicUpdate();
});

// 사이드바 토글
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
}

// 섹션 표시
function showSection(sectionId) {
    // 모든 섹션 숨기기
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('d-none');
    });
    
    // 선택된 섹션 표시
    document.getElementById(sectionId).classList.remove('d-none');
    
    // 네비게이션 링크 활성화 상태 업데이트
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.section === sectionId) {
            link.classList.add('active');
        }
    });
}

// 초기 데이터 로드
async function loadInitialData() {
    try {
        // 잔고 정보 로드
        const balanceResponse = await fetch('/api/balance');
        const balanceData = await balanceResponse.json();
        updateBalanceDisplay(balanceData);
        
        // 포지션 정보 로드
        const positionsResponse = await fetch('/api/positions');
        const positionsData = await positionsResponse.json();
        updatePositionsDisplay(positionsData);
        
        // 거래 내역 로드
        const tradesResponse = await fetch('/api/trades');
        const tradesData = await tradesResponse.json();
        updateTradesDisplay(tradesData);
        
        // 성과 지표 로드
        const performanceResponse = await fetch('/api/performance');
        const performanceData = await performanceResponse.json();
        updatePerformanceDisplay(performanceData);
        
        // 전략 매개변수 로드
        const strategyResponse = await fetch('/api/strategy_params');
        const strategyData = await strategyResponse.json();
        updateStrategyForm(strategyData);
        
        // 알림 설정 로드
        const notificationResponse = await fetch('/api/notifications');
        const notificationData = await notificationResponse.json();
        updateNotificationForm(notificationData);
        
    } catch (error) {
        console.error('데이터 로드 중 오류 발생:', error);
    }
}

// 주기적 업데이트 시작
function startPeriodicUpdate() {
    // 5초마다 데이터 업데이트
    updateInterval = setInterval(loadInitialData, 5000);
}

// 잔고 정보 업데이트
function updateBalanceDisplay(data) {
    if (data.error) {
        console.error('잔고 정보 조회 실패:', data.error);
        return;
    }
    
    const balance = data.balance;
    document.getElementById('totalBalance').textContent = balance.total.toFixed(2);
}

// 포지션 정보 업데이트
function updatePositionsDisplay(data) {
    if (data.error) {
        console.error('포지션 정보 조회 실패:', data.error);
        return;
    }
    
    const positions = data.positions;
    const tbody = document.getElementById('positionsTable');
    tbody.innerHTML = '';
    
    positions.forEach(position => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.symbol}</td>
            <td>${position.direction}</td>
            <td>${position.quantity.toFixed(4)}</td>
            <td>${position.entry_price.toFixed(2)}</td>
            <td>${position.current_price.toFixed(2)}</td>
            <td class="${position.pnl >= 0 ? 'positive' : 'negative'}">
                ${(position.pnl / position.entry_price * 100).toFixed(2)}%
            </td>
            <td class="${position.pnl >= 0 ? 'positive' : 'negative'}">
                ${position.pnl.toFixed(2)}
            </td>
        `;
        tbody.appendChild(row);
    });
    
    // 활성 포지션 수 업데이트
    document.getElementById('activePositions').textContent = positions.length;
}

// 거래 내역 업데이트
function updateTradesDisplay(data) {
    if (data.error) {
        console.error('거래 내역 조회 실패:', data.error);
        return;
    }
    
    const trades = data.trades;
    
    // 거래 내역 테이블 업데이트
    const tbody = document.getElementById('tradesTable');
    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(trade.timestamp).toLocaleString()}</td>
            <td>${trade.symbol}</td>
            <td>${trade.direction}</td>
            <td>${trade.price.toFixed(2)}</td>
            <td>${trade.quantity.toFixed(4)}</td>
            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
                ${(trade.pnl / trade.entry_price * 100).toFixed(2)}%
            </td>
            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
                ${trade.pnl.toFixed(2)}
            </td>
        `;
        tbody.appendChild(row);
    });
    
    // 최근 거래 목록 업데이트
    const recentTrades = trades.slice(-5).reverse();
    const recentTbody = document.getElementById('recentTrades');
    recentTbody.innerHTML = '';
    
    recentTrades.forEach(trade => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(trade.timestamp).toLocaleString()}</td>
            <td>${trade.symbol}</td>
            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
                ${(trade.pnl / trade.entry_price * 100).toFixed(2)}%
            </td>
        `;
        recentTbody.appendChild(row);
    });
}

// 성과 지표 업데이트
function updatePerformanceDisplay(data) {
    if (data.error) {
        console.error('성과 지표 조회 실패:', data.error);
        return;
    }
    
    const metrics = data.metrics;
    if (!metrics) return;
    
    // 성과 지표 업데이트
    document.getElementById('dailyReturn').textContent = `${(metrics.daily_return * 100).toFixed(2)}%`;
    document.getElementById('winRate').textContent = `${(metrics.win_rate * 100).toFixed(2)}%`;
    document.getElementById('sharpeRatio').textContent = metrics.sharpe_ratio.toFixed(2);
    document.getElementById('sortinoRatio').textContent = metrics.sortino_ratio.toFixed(2);
    document.getElementById('maxDrawdown').textContent = `${(metrics.max_drawdown * 100).toFixed(2)}%`;
    document.getElementById('profitFactor').textContent = metrics.profit_factor.toFixed(2);
    
    // 차트 업데이트
    updateCharts(data);
}

// 차트 업데이트
function updateCharts(data) {
    // 자본금 곡선 차트
    const equityChart = document.getElementById('equityChart');
    if (data.equity_curve) {
        const trace = {
            x: data.equity_curve.index,
            y: data.equity_curve.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Equity'
        };
        
        const layout = {
            title: '자본금 곡선',
            xaxis: { title: '날짜' },
            yaxis: { title: '자본금' },
            template: 'plotly_dark'
        };
        
        Plotly.newPlot(equityChart, [trace], layout);
    }
    
    // 수익률 분포 차트
    const returnsChart = document.getElementById('returnsChart');
    if (data.returns_distribution) {
        const trace = {
            x: data.returns_distribution.values,
            type: 'histogram',
            name: 'Returns'
        };
        
        const layout = {
            title: '수익률 분포',
            xaxis: { title: '수익률' },
            yaxis: { title: '빈도' },
            template: 'plotly_dark'
        };
        
        Plotly.newPlot(returnsChart, [trace], layout);
    }
    
    // 낙폭 차트
    const drawdownChart = document.getElementById('drawdownChart');
    if (data.drawdowns) {
        const trace = {
            x: data.drawdowns.index,
            y: data.drawdowns.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Drawdown',
            fill: 'tozeroy'
        };
        
        const layout = {
            title: '낙폭',
            xaxis: { title: '날짜' },
            yaxis: { title: '낙폭' },
            template: 'plotly_dark'
        };
        
        Plotly.newPlot(drawdownChart, [trace], layout);
    }
}

// 전략 매개변수 폼 업데이트
function updateStrategyForm(data) {
    if (data.error) {
        console.error('전략 매개변수 조회 실패:', data.error);
        return;
    }
    
    const params = data.params;
    const form = document.getElementById('strategyForm');
    
    // 폼 필드 업데이트
    Object.keys(params).forEach(key => {
        const input = form.elements[key];
        if (input) {
            input.value = params[key];
        }
    });
}

// 알림 설정 폼 업데이트
function updateNotificationForm(data) {
    if (data.error) {
        console.error('알림 설정 조회 실패:', data.error);
        return;
    }
    
    const settings = data.settings;
    const form = document.getElementById('notificationForm');
    
    // 폼 필드 업데이트
    Object.keys(settings).forEach(key => {
        const input = form.elements[key];
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = settings[key];
            } else {
                input.value = settings[key];
            }
        }
    });
}

// 전략 매개변수 제출 처리
async function handleStrategySubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const params = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/api/strategy_params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        alert('전략 매개변수가 업데이트되었습니다.');
        
    } catch (error) {
        console.error('전략 매개변수 업데이트 실패:', error);
        alert('전략 매개변수 업데이트에 실패했습니다.');
    }
}

// 알림 설정 제출 처리
async function handleNotificationSubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const settings = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/api/notifications', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        alert('알림 설정이 업데이트되었습니다.');
        
    } catch (error) {
        console.error('알림 설정 업데이트 실패:', error);
        alert('알림 설정 업데이트에 실패했습니다.');
    }
} 