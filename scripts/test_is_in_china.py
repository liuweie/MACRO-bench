import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from user_simulator.travel_simulator import TravelUserSimulator

sim = TravelUserSimulator()

tests = ['東京', '东京', 'Tokyo', '大阪', '上海', 'Beijing', '서울', 'Seoul', '日本', '中国', '广州市', 'Osaka', 'New York']
for t in tests:
    print(t, '->', sim._is_in_china(t))
