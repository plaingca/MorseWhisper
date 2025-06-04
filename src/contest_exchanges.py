"""
Amateur Radio Contest Exchange Generator

Generates typical contest exchanges for various amateur radio contests:
- RST reports (signal reports)
- Grid squares (Maidenhead locator system)
- Serial numbers
- Zones (CQ zones, ITU zones)
- States/provinces
- Power categories
- ARRL sections
"""

import random
import string
from typing import Dict, List, Optional, Tuple


class ContestExchangeGenerator:
    """Generate realistic amateur radio contest exchanges."""
    
    # Grid square components
    GRID_FIELDS = 'ABCDEFGHIJKLMNOPQRS'  # First two letters
    GRID_SQUARES = '0123456789'  # Two numbers
    GRID_SUBSQUARES = 'abcdefghijklmnopqrstuvwx'  # Optional last two letters
    
    # US states for domestic contests
    US_STATES = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC'  # District of Columbia
    ]
    
    # Canadian provinces
    CANADIAN_PROVINCES = [
        'BC', 'AB', 'SK', 'MB', 'ON', 'QC', 'NB', 'NS', 'PE', 'NL',
        'YT', 'NT', 'NU'
    ]
    
    # ARRL/RAC sections for Sweepstakes
    ARRL_SECTIONS = [
        # US Sections
        'CT', 'EMA', 'ME', 'NH', 'RI', 'VT', 'WMA', 'ENY', 'NLI', 'NNJ',
        'NNY', 'SNJ', 'WNY', 'DE', 'EPA', 'MDC', 'WPA', 'AL', 'GA', 'KY',
        'NC', 'NFL', 'SC', 'SFL', 'WCF', 'TN', 'VA', 'PR', 'VI', 'AR',
        'LA', 'MS', 'NM', 'NTX', 'OK', 'STX', 'WTX', 'EB', 'LAX', 'ORG',
        'SB', 'SCV', 'SDG', 'SF', 'SJV', 'SV', 'PAC', 'AZ', 'EWA', 'ID',
        'MT', 'NV', 'OR', 'UT', 'WWA', 'WY', 'AK', 'MI', 'OH', 'WV', 'IL',
        'IN', 'WI', 'CO', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD',
        # Canadian Sections
        'BC', 'AB', 'SK', 'MB', 'ON', 'GTA', 'QC', 'MAR', 'NL'
    ]
    
    # CQ Zones (1-40)
    CQ_ZONES = list(range(1, 41))
    
    # ITU Zones
    ITU_ZONES = list(range(1, 91))
    
    # Power categories
    POWER_CATEGORIES = ['QRP', 'LP', 'HP']  # 5W, 100W, 1500W
    
    # Common contest types and their exchanges
    CONTEST_FORMATS = {
        'CQWW': {
            'exchange': ['rst', 'cq_zone'],
            'description': 'CQ World Wide DX Contest'
        },
        'CQWPX': {
            'exchange': ['rst', 'serial'],
            'description': 'CQ WPX Contest'
        },
        'ARRLDX': {
            'exchange_dx': ['rst', 'power'],
            'exchange_us': ['rst', 'state'],
            'description': 'ARRL International DX Contest'
        },
        'SWEEPSTAKES': {
            'exchange': ['serial', 'precedence', 'check', 'section'],
            'description': 'ARRL Sweepstakes'
        },
        'FIELD_DAY': {
            'exchange': ['class', 'section'],
            'description': 'ARRL Field Day'
        },
        'NAQP': {
            'exchange': ['name', 'state'],
            'description': 'North American QSO Party'
        },
        'SPRINT': {
            'exchange': ['serial', 'name', 'state'],
            'description': 'NA Sprint'
        },
        'IARU': {
            'exchange': ['rst', 'itu_zone'],
            'description': 'IARU HF Championship'
        },
        'WAE': {
            'exchange': ['rst', 'serial'],
            'description': 'Worked All Europe'
        },
        'ALL_ASIAN': {
            'exchange': ['rst', 'age'],
            'description': 'All Asian DX Contest'
        }
    }
    
    # Common first names for contests
    FIRST_NAMES = [
        'JOHN', 'MIKE', 'DAVE', 'BOB', 'JIM', 'TOM', 'BILL', 'STEVE',
        'RICK', 'PAUL', 'MARK', 'GARY', 'FRED', 'CARL', 'DAN', 'ED',
        'AL', 'ART', 'BEN', 'CHUCK', 'DICK', 'FRANK', 'GEORGE', 'JACK',
        'JOE', 'KEN', 'LARRY', 'LEE', 'PETE', 'RAY', 'RON', 'SAM',
        'MARY', 'PAT', 'SUE', 'ANN', 'LIZ', 'JANE', 'KATE', 'LYNN'
    ]
    
    def __init__(self):
        """Initialize the exchange generator."""
        pass
    
    def generate_rst(self, mode: str = 'CW') -> str:
        """
        Generate RST report.
        
        Args:
            mode: 'CW' or 'PHONE'
            
        Returns:
            RST report (e.g., '599' for CW, '59' for phone)
        """
        if mode.upper() == 'CW':
            # Readability (1-5), Strength (1-9), Tone (1-9)
            # Contest exchanges are typically 599 or occasionally 579, 589
            if random.random() < 0.9:
                return '599'
            else:
                return random.choice(['579', '589', '569', '559'])
        else:  # Phone
            # Readability (1-5), Strength (1-9)
            if random.random() < 0.9:
                return '59'
            else:
                return random.choice(['57', '58', '56', '55'])
    
    def generate_grid_square(self, precision: int = 4) -> str:
        """
        Generate Maidenhead grid square.
        
        Args:
            precision: 4 or 6 characters
            
        Returns:
            Grid square (e.g., 'FN31' or 'FN31pr')
        """
        # Field (2 letters)
        field = random.choice(self.GRID_FIELDS) + random.choice(self.GRID_FIELDS)
        
        # Square (2 numbers)
        square = random.choice(self.GRID_SQUARES) + random.choice(self.GRID_SQUARES)
        
        grid = field + square
        
        # Optional subsquare (2 letters)
        if precision == 6:
            subsquare = random.choice(self.GRID_SUBSQUARES) + random.choice(self.GRID_SUBSQUARES)
            grid += subsquare
        
        return grid
    
    def generate_serial(self, start: int = 1, end: int = 9999) -> str:
        """Generate contest serial number."""
        serial = random.randint(start, end)
        
        # In CW, often send abbreviated numbers
        if random.random() < 0.3 and serial >= 1000:
            # Abbreviate thousands (e.g., 1234 -> 1T34)
            thousands = serial // 1000
            remainder = serial % 1000
            if thousands == 1:
                return f"T{remainder:03d}"
            elif thousands in range(2, 10):
                return f"{thousands}T{remainder:03d}"
        
        # Sometimes leading zeros are sent
        if serial < 100 and random.random() < 0.5:
            return f"{serial:03d}"
        
        return str(serial)
    
    def generate_zone(self, zone_type: str = 'CQ') -> str:
        """Generate CQ or ITU zone."""
        if zone_type.upper() == 'CQ':
            return str(random.choice(self.CQ_ZONES))
        else:  # ITU
            return str(random.choice(self.ITU_ZONES))
    
    def generate_state_province(self, country: str = 'USA') -> str:
        """Generate state or province abbreviation."""
        if country.upper() == 'USA':
            return random.choice(self.US_STATES)
        elif country.upper() == 'CANADA':
            return random.choice(self.CANADIAN_PROVINCES)
        else:
            # Mix of US and Canadian
            all_regions = self.US_STATES + self.CANADIAN_PROVINCES
            return random.choice(all_regions)
    
    def generate_section(self) -> str:
        """Generate ARRL/RAC section."""
        return random.choice(self.ARRL_SECTIONS)
    
    def generate_power(self) -> str:
        """Generate power level for ARRL DX contest."""
        # Most stations are high power
        weights = [5, 20, 75]  # QRP, LP, HP weights
        power_num = random.choices([5, 100, 1500], weights=weights)[0]
        
        # In contests, often just send the number or abbreviation
        if random.random() < 0.5:
            return str(power_num)
        else:
            if power_num == 5:
                return 'QRP'
            elif power_num == 100:
                return '100'  # LP is less commonly abbreviated
            else:
                return 'KW'  # 1KW or more
    
    def generate_name(self) -> str:
        """Generate operator first name."""
        return random.choice(self.FIRST_NAMES)
    
    def generate_check(self) -> str:
        """Generate check (year first licensed) for Sweepstakes."""
        # Year range from 1920s to current
        current_year = 2024
        year = random.randint(1925, current_year)
        
        # Return last 2 digits
        return str(year % 100).zfill(2)
    
    def generate_precedence(self) -> str:
        """Generate precedence for Sweepstakes."""
        # A=Low Power, B=High Power, Q=QRP, M=Multi-op, S=School
        precedences = ['A', 'B', 'Q', 'M', 'S']
        weights = [35, 50, 5, 8, 2]  # Rough distribution
        return random.choices(precedences, weights=weights)[0]
    
    def generate_field_day_class(self) -> str:
        """Generate Field Day class (e.g., '3A', '1B', '2F')."""
        # Number of transmitters
        transmitters = random.choices(
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            weights=[20, 25, 20, 15, 10, 5, 2, 1, 1, 1]
        )[0]
        
        # Category
        categories = ['A', 'B', 'C', 'D', 'E', 'F']
        category_weights = [60, 20, 5, 5, 5, 5]
        category = random.choices(categories, weights=category_weights)[0]
        
        return f"{transmitters}{category}"
    
    def generate_age(self) -> str:
        """Generate age for All Asian DX Contest."""
        # Operator age, typically 20-80
        age = random.randint(20, 80)
        
        # Sometimes send as two digits even if under 10 (rare)
        if age < 10:
            return f"0{age}"
        return str(age)
    
    def generate_exchange(self, contest_type: str, 
                         station_info: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate complete contest exchange.
        
        Args:
            contest_type: Type of contest (from CONTEST_FORMATS)
            station_info: Optional dict with station details (country, etc.)
            
        Returns:
            Dictionary with exchange elements
        """
        if contest_type not in self.CONTEST_FORMATS:
            raise ValueError(f"Unknown contest type: {contest_type}")
        
        contest = self.CONTEST_FORMATS[contest_type]
        exchange = {}
        
        # Handle contests with different exchanges for DX/US stations
        if contest_type == 'ARRLDX':
            if station_info and station_info.get('country') == 'USA':
                exchange_format = contest['exchange_us']
            else:
                exchange_format = contest['exchange_dx']
        else:
            exchange_format = contest.get('exchange', [])
        
        # Generate each element
        for element in exchange_format:
            if element == 'rst':
                exchange['rst'] = self.generate_rst()
            elif element == 'serial':
                exchange['serial'] = self.generate_serial()
            elif element == 'cq_zone':
                exchange['zone'] = self.generate_zone('CQ')
            elif element == 'itu_zone':
                exchange['zone'] = self.generate_zone('ITU')
            elif element == 'state':
                exchange['state'] = self.generate_state_province()
            elif element == 'power':
                exchange['power'] = self.generate_power()
            elif element == 'name':
                exchange['name'] = self.generate_name()
            elif element == 'section':
                exchange['section'] = self.generate_section()
            elif element == 'check':
                exchange['check'] = self.generate_check()
            elif element == 'precedence':
                exchange['precedence'] = self.generate_precedence()
            elif element == 'class':
                exchange['class'] = self.generate_field_day_class()
            elif element == 'age':
                exchange['age'] = self.generate_age()
        
        return exchange
    
    def format_exchange_string(self, exchange: Dict[str, str], 
                              contest_type: str) -> str:
        """
        Format exchange dictionary as typical contest string.
        
        Args:
            exchange: Exchange dictionary
            contest_type: Contest type
            
        Returns:
            Formatted exchange string
        """
        if contest_type == 'CQWW':
            return f"{exchange['rst']} {exchange['zone']}"
        elif contest_type == 'CQWPX':
            return f"{exchange['rst']} {exchange['serial']}"
        elif contest_type == 'ARRLDX':
            if 'power' in exchange:
                return f"{exchange['rst']} {exchange['power']}"
            else:
                return f"{exchange['rst']} {exchange['state']}"
        elif contest_type == 'SWEEPSTAKES':
            # Format: Serial Precedence Call Check Section
            # (Call is added by the sender)
            return f"{exchange['serial']} {exchange['precedence']} {exchange['check']} {exchange['section']}"
        elif contest_type == 'FIELD_DAY':
            return f"{exchange['class']} {exchange['section']}"
        elif contest_type == 'NAQP':
            return f"{exchange['name']} {exchange['state']}"
        elif contest_type == 'SPRINT':
            return f"{exchange['serial']} {exchange['name']} {exchange['state']}"
        else:
            # Generic format
            return ' '.join(exchange.values())


def generate_samples():
    """Generate sample contest exchanges."""
    generator = ContestExchangeGenerator()
    
    print("Sample contest exchanges:\n")
    
    for contest_type in ['CQWW', 'CQWPX', 'ARRLDX', 'SWEEPSTAKES', 'FIELD_DAY', 'NAQP']:
        print(f"{contest_type}:")
        for i in range(3):
            exchange = generator.generate_exchange(contest_type)
            formatted = generator.format_exchange_string(exchange, contest_type)
            print(f"  {formatted}")
        print()
    
    print("Grid squares:")
    for _ in range(5):
        print(f"  {generator.generate_grid_square()}")


if __name__ == "__main__":
    generate_samples() 