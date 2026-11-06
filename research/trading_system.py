"""
实时交易系统
- 订单管理
- 实时交易接口
- 实时风险控制
- 交易状态监控
- 交易成本计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import json
import logging
from abc import ABC, abstractmethod
from queue import Queue, Empty
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"      # 待发送
    SUBMITTED = "SUBMITTED"  # 已提交
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"       # 完全成交
    CANCELLED = "CANCELLED" # 已取消
    REJECTED = "REJECTED"   # 已拒绝
    EXPIRED = "EXPIRED"     # 已过期


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"        # 市价单
    LIMIT = "LIMIT"         # 限价单
    STOP = "STOP"           # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """订单有效期"""
    DAY = "DAY"              # 当日有效
    GTC = "GTC"             # 撤销前有效
    IOC = "IOC"             # 立即成交或撤销
    FOK = "FOK"             # 全部成交或撤销


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    commission: float = 0.0
    message: str = ""
    parent_id: Optional[str] = None  # 用于关联订单（如拆单）
    strategy_id: Optional[str] = None  # 策略ID
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'message': self.message,
            'parent_id': self.parent_id,
            'strategy_id': self.strategy_id,
            'tags': self.tags,
        }

    def update_status(self, new_status: OrderStatus, message: str = ""):
        """更新订单状态"""
        self.status = new_status
        self.updated_at = datetime.now()
        self.message = message
        logger.info(f"订单 {self.order_id} 状态更新: {new_status.value} - {message}")


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    strategy_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
        }


class TradingInterface(ABC):
    """交易接口抽象类"""
    
    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        pass


class MockTradingInterface(TradingInterface):
    """模拟交易接口（用于测试）"""
    
    def __init__(self):
        self.orders = {}
        self.trades = []
        self.account_balance = 1000000.0
        self.positions = {}
        
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        # 模拟订单处理延迟
        time.sleep(0.1)
        
        # 模拟订单验证
        if order.order_type == OrderType.LIMIT and order.price is None:
            order.update_status(OrderStatus.REJECTED, "限价单缺少价格")
            return False
        
        if order.quantity <= 0:
            order.update_status(OrderStatus.REJECTED, "数量必须大于0")
            return False
        
        # 模拟订单提交
        order.update_status(OrderStatus.SUBMITTED)
        self.orders[order.order_id] = order
        
        # 模拟成交
        self._simulate_fill(order)
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.update_status(OrderStatus.CANCELLED, "用户取消")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return {
            'balance': self.account_balance,
            'positions': self.positions,
            'available_cash': self.account_balance - sum(
                pos['quantity'] * pos['current_price'] for pos in self.positions.values()
            ),
        }
    
    def get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        # 模拟市场数据
        return {
            'symbol': symbol,
            'price': 100.0 + np.random.normal(0, 1),
            'volume': np.random.randint(10000, 100000),
            'timestamp': datetime.now().isoformat(),
        }
    
    def _simulate_fill(self, order: Order):
        """模拟成交"""
        # 模拟部分成交
        fill_ratio = np.random.uniform(0.8, 1.0)
        filled_quantity = int(order.quantity * fill_ratio)
        
        # 模拟成交价格
        if order.order_type == OrderType.MARKET:
            filled_price = self.get_market_data(order.symbol)['price']
        else:
            filled_price = order.price
        
        # 更新订单状态
        if filled_quantity == order.quantity:
            order.update_status(OrderStatus.FILLED)
        else:
            order.update_status(OrderStatus.PARTIALLY_FILLED)
            order.filled_quantity = filled_quantity
            order.filled_price = filled_price
        
        # 创建成交记录
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=filled_quantity,
            price=filled_price,
            commission=filled_quantity * filled_price * 0.0003,  # 手续费
            timestamp=datetime.now(),
            strategy_id=order.strategy_id,
        )
        self.trades.append(trade)
        
        # 更新账户
        if order.side == OrderSide.BUY:
            cost = filled_quantity * filled_price + trade.commission
            self.account_balance -= cost
            if order.symbol not in self.positions:
                self.positions[order.symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'current_price': filled_price,
                }
            pos = self.positions[order.symbol]
            total_quantity = pos['quantity'] + filled_quantity
            pos['avg_price'] = (pos['quantity'] * pos['avg_price'] + filled_quantity * filled_price) / total_quantity
            pos['quantity'] = total_quantity
            pos['current_price'] = filled_price
        else:
            revenue = filled_quantity * filled_price - trade.commission
            self.account_balance += revenue
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos['quantity'] -= filled_quantity
                if pos['quantity'] == 0:
                    del self.positions[order.symbol]


class RealTimeRiskManager:
    """实时风险管理器"""
    
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.05):
        self.max_position_size = max_position_size  # 单个持仓最大比例
        self.max_daily_loss = max_daily_loss      # 最大日亏损比例
        self.daily_pnl = 0.0
        self.position_sizes = {}
        self.order_history = []
        
    def check_order(self, order: Order, account_info: Dict) -> Tuple[bool, str]:
        """检查订单风险"""
        # 检查日亏损限制
        if self.daily_pnl < -account_info['balance'] * self.max_daily_loss:
            return False, f"超过最大日亏损限制 ({self.max_daily_loss:.1%})"
        
        # 检查持仓限制
        current_position = self.position_sizes.get(order.symbol, 0)
        new_position = current_position + order.quantity if order.side == OrderSide.BUY else current_position - order.quantity
        
        if abs(new_position) / account_info['balance'] > self.max_position_size:
            return False, f"超过持仓限制 ({self.max_position_size:.1%})"
        
        return True, "风险检查通过"
    
    def update_position(self, symbol: str, quantity: int):
        """更新持仓"""
        self.position_sizes[symbol] = self.position_sizes.get(symbol, 0) + quantity
    
    def update_pnl(self, pnl: float):
        """更新盈亏"""
        self.daily_pnl += pnl


class OrderManager:
    """订单管理器"""
    
    def __init__(self, trading_interface: TradingInterface, 
                 risk_manager: RealTimeRiskManager):
        self.trading_interface = trading_interface
        self.risk_manager = risk_manager
        self.orders = {}
        self.order_queue = Queue()
        self.order_history = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def start(self):
        """启动订单管理器"""
        self.is_running = True
        self.executor.submit(self._process_orders)
        logger.info("订单管理器已启动")
    
    def stop(self):
        """停止订单管理器"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("订单管理器已停止")
    
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        # 风险检查
        account_info = self.trading_interface.get_account_info()
        risk_check, risk_message = self.risk_manager.check_order(order, account_info)
        
        if not risk_check:
            order.update_status(OrderStatus.REJECTED, risk_message)
            self.order_history.append(order)
            return False
        
        # 加入订单队列
        self.order_queue.put(order)
        self.orders[order.order_id] = order
        logger.info(f"订单已加入队列: {order.order_id}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            return False
        
        # 尝试取消订单
        success = self.trading_interface.cancel_order(order_id)
        if success:
            order.update_status(OrderStatus.CANCELLED, "系统取消")
        
        return success
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def get_all_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取所有订单"""
        orders = list(self.orders.values())
        if status:
            orders = [order for order in orders if order.status == status]
        return orders
    
    def _process_orders(self):
        """处理订单队列"""
        while self.is_running:
            try:
                # 从队列获取订单
                order = self.order_queue.get(timeout=1.0)
                
                # 提交订单
                success = self.trading_interface.submit_order(order)
                if success:
                    logger.info(f"订单提交成功: {order.order_id}")
                else:
                    logger.error(f"订单提交失败: {order.order_id}")
                
                # 标记任务完成
                self.order_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理订单时发生错误: {e}")


class RealTimeTradingSystem:
    """实时交易系统"""
    
    def __init__(self, trading_interface: TradingInterface):
        self.trading_interface = trading_interface
        self.order_manager = OrderManager(trading_interface, RealTimeRiskManager())
        self.active_strategies = {}
        self.market_data = {}
        self.is_running = False
        
    def start(self):
        """启动交易系统"""
        self.is_running = True
        self.order_manager.start()
        logger.info("实时交易系统已启动")
        
        # 启动市场数据监控
        self._start_market_monitor()
    
    def stop(self):
        """停止交易系统"""
        self.is_running = False
        self.order_manager.stop()
        logger.info("实时交易系统已停止")
    
    def add_strategy(self, strategy_id: str, strategy):
        """添加交易策略"""
        self.active_strategies[strategy_id] = strategy
        logger.info(f"策略已添加: {strategy_id}")
    
    def remove_strategy(self, strategy_id: str):
        """移除交易策略"""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"策略已移除: {strategy_id}")
    
    def place_order(self, strategy_id: str, symbol: str, side: OrderSide, 
                   quantity: int, order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None, **kwargs) -> str:
        """下单"""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            strategy_id=strategy_id,
            **kwargs
        )
        
        success = self.order_manager.submit_order(order)
        return order.order_id if success else None
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        return self.order_manager.cancel_order(order_id)
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.order_manager.get_order_status(order_id)
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return self.trading_interface.get_account_info()
    
    def get_strategy_performance(self, strategy_id: str) -> Dict:
        """获取策略表现"""
        strategy_orders = [order for order in self.order_manager.get_all_orders() 
                          if order.strategy_id == strategy_id]
        
        if not strategy_orders:
            return {'error': '策略无交易记录'}
        
        # 计算策略表现
        total_trades = len(strategy_orders)
        winning_trades = len([o for o in strategy_orders if o.status == OrderStatus.FILLED])
        
        # 计算盈亏
        total_pnl = 0.0
        for order in strategy_orders:
            if order.status == OrderStatus.FILLED and order.filled_price:
                if order.side == OrderSide.BUY:
                    cost = order.filled_quantity * order.filled_price + order.commission
                    total_pnl -= cost
                else:
                    revenue = order.filled_quantity * order.filled_price - order.commission
                    total_pnl += revenue
        
        return {
            'strategy_id': strategy_id,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_commission': np.mean([o.commission for o in strategy_orders]) if strategy_orders else 0,
        }
    
    def _start_market_monitor(self):
        """启动市场监控"""
        def monitor():
            while self.is_running:
                try:
                    # 更新市场数据
                    for strategy_id, strategy in self.active_strategies.items():
                        # 这里应该调用策略的实时分析方法
                        pass
                    
                    time.sleep(1.0)  # 每秒更新一次
                    
                except Exception as e:
                    logger.error(f"市场监控错误: {e}")
        
        import threading
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def generate_trading_report(self) -> Dict:
        """生成交易报告"""
        account_info = self.get_account_info()
        
        # 统计订单
        all_orders = self.order_manager.get_all_orders()
        order_stats = {}
        for status in OrderStatus:
            order_stats[status.value] = len([o for o in all_orders if o.status == status])
        
        # 策略表现
        strategy_performance = {}
        for strategy_id in self.active_strategies.keys():
            perf = self.get_strategy_performance(strategy_id)
            strategy_performance[strategy_id] = perf
        
        return {
            'account_info': account_info,
            'order_statistics': order_stats,
            'strategy_performance': strategy_performance,
            'total_orders': len(all_orders),
            'active_strategies': len(self.active_strategies),
            'timestamp': datetime.now().isoformat(),
        }


class TradingAPI:
    """交易API接口"""
    
    def __init__(self, trading_system: RealTimeTradingSystem):
        self.trading_system = trading_system
    
    def submit_order(self, order_data: Dict) -> Dict:
        """提交订单API"""
        try:
            order = Order(
                order_id=order_data.get('order_id', str(uuid.uuid4())),
                symbol=order_data['symbol'],
                side=OrderSide(order_data['side']),
                order_type=OrderType(order_data['order_type']),
                quantity=order_data['quantity'],
                price=order_data.get('price'),
                stop_price=order_data.get('stop_price'),
                time_in_force=TimeInForce(order_data.get('time_in_force', 'DAY')),
                strategy_id=order_data.get('strategy_id'),
                tags=order_data.get('tags', {}),
            )
            
            success = self.trading_system.place_order(
                strategy_id=order_data.get('strategy_id', 'default'),
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                tags=order.tags,
            )
            
            return {
                'success': success,
                'order_id': order.order_id if success else None,
                'message': '订单提交成功' if success else '订单提交失败',
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'订单提交错误: {str(e)}',
            }
    
    def get_orders(self, strategy_id: Optional[str] = None, 
                  status: Optional[str] = None) -> Dict:
        """获取订单列表"""
        try:
            if strategy_id:
                orders = [o for o in self.trading_system.order_manager.get_all_orders() 
                         if o.strategy_id == strategy_id]
            else:
                orders = self.trading_system.order_manager.get_all_orders()
            
            if status:
                orders = [o for o in orders if o.status.value == status]
            
            return {
                'success': True,
                'orders': [order.to_dict() for order in orders],
                'count': len(orders),
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'获取订单错误: {str(e)}',
            }
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        try:
            account_info = self.trading_system.get_account_info()
            return {
                'success': True,
                'account_info': account_info,
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'获取账户信息错误: {str(e)}',
            }
    
    def get_strategy_performance(self, strategy_id: str) -> Dict:
        """获取策略表现"""
        try:
            performance = self.trading_system.get_strategy_performance(strategy_id)
            return {
                'success': True,
                'performance': performance,
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'获取策略表现错误: {str(e)}',
            }