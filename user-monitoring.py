#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROFESSIONAL TELEGRAM USERNAME HUNTER
Hevorix Edition - Advanced Username Monitoring & Acquisition System
"""

import asyncio
import aiofiles
import logging
import sys
import json
import random
import string
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import pickle

from telethon import TelegramClient, events, functions, types
from telethon.errors import (
    UsernameNotOccupiedError,
    UsernameInvalidError,
    FloodWaitError,
    UsernameOccupiedError,
    ChatAdminRequiredError,
    ChannelPrivateError,
    AuthKeyError
)
from telethon.tl.functions.channels import (
    CreateChannelRequest,
    UpdateUsernameRequest,
    GetFullChannelRequest
)
from telethon.tl.functions.account import UpdateUsernameRequest as UpdateAccountUsername
from telethon.tl.types import (
    Channel,
    User,
    Chat
)

# ========== CONFIGURATION ==========
class Config:
    # API Credentials
    API_ID = 28314907
    API_HASH = '01bffe95d8af217cf794e66d989c227e'
    PHONE_NUMBER = '+998774490625'
    
    # Target Configuration
    TARGET_USERNAMES = ['algorythm', 'algorithm', 'algo', 'trade', 'crypto', 'hack']  # Multiple targets
    USERNAME_TYPES = ['channel', 'account']  # What to claim for
    PRIORITY_USERNAMES = ['algorythm']  # High priority targets
    
    # Monitoring Settings
    CHECK_INTERVAL = 60  # seconds between checks
    QUICK_CHECK_INTERVAL = 5  # seconds for priority targets
    MAX_RETRIES = 5
    RETRY_DELAY = 30  # seconds
    
    # Channel Creation Settings
    CHANNEL_TITLE_PREFIX = "Official "
    CHANNEL_ABOUT = "Verified channel"
    CHANNEL_PRIVATE = False
    
    # Account Settings (if claiming for account)
    ACCOUNT_FIRST_NAME = "Algorithm"
    ACCOUNT_LAST_NAME = "Master"
    ACCOUNT_BIO = "Official account"
    
    # Performance Settings
    MAX_CONCURRENT_CHECKS = 3
    USE_PROXY = False
    PROXY_CONFIG = None  # (protocol, host, port, username, password)
    
    # Storage
    SESSION_DIR = "hunter_sessions"
    LOG_FILE = "username_hunter.log"
    DATA_FILE = "hunter_data.json"
    BACKUP_DIR = "backups"
    
    # Alert System
    ENABLE_ALERTS = True
    ALERT_METHODS = ['log', 'file', 'telegram']  # log, file, telegram, email
    TELEGRAM_ALERT_CHAT_ID = None  # Chat ID for alerts
    ALERT_COOLDOWN = 300  # seconds between duplicate alerts
    
    # Security
    ENCRYPT_DATA = False
    ROTATE_SESSIONS = True
    SESSION_ROTATION_HOURS = 24

# ========== LOGGING SETUP ==========
class LoggerSetup:
    @staticmethod
    def setup_logger(name: str = "UsernameHunter"):
        """Setup comprehensive logging system"""
        # Create log directory
        Path(Config.LOG_FILE).parent.mkdir(exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

logger = LoggerSetup.setup_logger()

# ========== DATA MODELS ==========
class UsernameType(Enum):
    CHANNEL = "channel"
    ACCOUNT = "account"
    BOT = "bot"

@dataclass
class TargetUsername:
    username: str
    username_type: UsernameType
    priority: int = 1
    last_checked: Optional[datetime] = None
    status: str = "unknown"
    owner_id: Optional[int] = None
    owner_username: Optional[str] = None
    last_active: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def is_available(self) -> bool:
        return self.status == "available"
    
    def is_occupied(self) -> bool:
        return self.status == "occupied"

@dataclass
class AcquisitionResult:
    success: bool
    username: str
    username_type: UsernameType
    entity_id: Optional[int] = None
    entity_username: Optional[str] = None
    timestamp: datetime = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class UsernameDatabase:
    """Persistent storage for username tracking"""
    
    def __init__(self, data_file: str = Config.DATA_FILE):
        self.data_file = data_file
        self.targets: Dict[str, TargetUsername] = {}
        self.acquisitions: List[AcquisitionResult] = []
        self.attempts: Dict[str, List[datetime]] = {}
        self.load_data()
    
    def load_data(self):
        """Load data from file"""
        try:
            if Path(self.data_file).exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load targets
                    for username, target_data in data.get('targets', {}).items():
                        target = TargetUsername(
                            username=target_data['username'],
                            username_type=UsernameType(target_data['username_type']),
                            priority=target_data.get('priority', 1),
                            last_checked=datetime.fromisoformat(target_data['last_checked']) if target_data.get('last_checked') else None,
                            status=target_data.get('status', 'unknown'),
                            owner_id=target_data.get('owner_id'),
                            owner_username=target_data.get('owner_username'),
                            last_active=datetime.fromisoformat(target_data['last_active']) if target_data.get('last_active') else None,
                            created_at=datetime.fromisoformat(target_data['created_at']) if target_data.get('created_at') else None
                        )
                        self.targets[username] = target
                    
                    # Load acquisitions
                    for acq_data in data.get('acquisitions', []):
                        acquisition = AcquisitionResult(
                            success=acq_data['success'],
                            username=acq_data['username'],
                            username_type=UsernameType(acq_data['username_type']),
                            entity_id=acq_data.get('entity_id'),
                            entity_username=acq_data.get('entity_username'),
                            timestamp=datetime.fromisoformat(acq_data['timestamp']),
                            error=acq_data.get('error')
                        )
                        self.acquisitions.append(acquisition)
                        
                logger.info(f"Loaded database with {len(self.targets)} targets")
            else:
                logger.info("No existing database found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
    
    def save_data(self):
        """Save data to file"""
        try:
            # Create backup
            self.create_backup()
            
            data = {
                'targets': {},
                'acquisitions': [],
                'last_updated': datetime.now().isoformat()
            }
            
            # Save targets
            for username, target in self.targets.items():
                data['targets'][username] = {
                    'username': target.username,
                    'username_type': target.username_type.value,
                    'priority': target.priority,
                    'last_checked': target.last_checked.isoformat() if target.last_checked else None,
                    'status': target.status,
                    'owner_id': target.owner_id,
                    'owner_username': target.owner_username,
                    'last_active': target.last_active.isoformat() if target.last_active else None,
                    'created_at': target.created_at.isoformat() if target.created_at else None
                }
            
            # Save acquisitions
            for acquisition in self.acquisitions:
                data['acquisitions'].append({
                    'success': acquisition.success,
                    'username': acquisition.username,
                    'username_type': acquisition.username_type.value,
                    'entity_id': acquisition.entity_id,
                    'entity_username': acquisition.entity_username,
                    'timestamp': acquisition.timestamp.isoformat(),
                    'error': acquisition.error
                })
            
            # Write to file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Database saved with {len(self.targets)} targets")
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def create_backup(self):
        """Create backup of data"""
        try:
            backup_dir = Path(Config.BACKUP_DIR)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"hunter_backup_{timestamp}.json"
            
            if Path(self.data_file).exists():
                import shutil
                shutil.copy2(self.data_file, backup_file)
                logger.debug(f"Created backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Backup failed: {e}")
    
    def add_target(self, target: TargetUsername):
        """Add target to database"""
        self.targets[target.username] = target
        self.save_data()
    
    def update_target(self, username: str, **kwargs):
        """Update target information"""
        if username in self.targets:
            target = self.targets[username]
            for key, value in kwargs.items():
                if hasattr(target, key):
                    setattr(target, key, value)
            target.last_checked = datetime.now()
            self.save_data()
    
    def add_acquisition(self, result: AcquisitionResult):
        """Add acquisition result"""
        self.acquisitions.append(result)
        self.save_data()
    
    def record_attempt(self, username: str):
        """Record attempt timestamp"""
        if username not in self.attempts:
            self.attempts[username] = []
        self.attempts[username].append(datetime.now())
        
        # Keep only last 100 attempts
        if len(self.attempts[username]) > 100:
            self.attempts[username] = self.attempts[username][-100:]
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        stats = {
            'total_targets': len(self.targets),
            'available': sum(1 for t in self.targets.values() if t.status == 'available'),
            'occupied': sum(1 for t in self.targets.values() if t.status == 'occupied'),
            'unknown': sum(1 for t in self.targets.values() if t.status == 'unknown'),
            'total_acquisitions': len([a for a in self.acquisitions if a.success]),
            'failed_acquisitions': len([a for a in self.acquisitions if not a.success]),
            'total_attempts': sum(len(v) for v in self.attempts.values())
        }
        return stats

# ========== USERNAME CHECKER ==========
class UsernameChecker:
    """Advanced username availability checker"""
    
    def __init__(self, client: TelegramClient, database: UsernameDatabase):
        self.client = client
        self.db = database
        self.checked_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.cache_duration = timedelta(minutes=5)
    
    async def check_username(self, username: str, username_type: UsernameType = UsernameType.CHANNEL) -> TargetUsername:
        """Check username availability"""
        try:
            # Check cache first
            if username in self.checked_cache:
                cached_result, cached_time = self.checked_cache[username]
                if datetime.now() - cached_time < self.cache_duration:
                    logger.debug(f"Cache hit for {username}")
                    return TargetUsername(
                        username=username,
                        username_type=username_type,
                        status="available" if cached_result else "occupied",
                        last_checked=datetime.now()
                    )
            
            # Record attempt
            self.db.record_attempt(username)
            
            # Try to resolve username
            try:
                entity = await self.client.get_entity(username)
                
                # Entity found - username is occupied
                status = "occupied"
                owner_id = entity.id
                owner_username = getattr(entity, 'username', None)
                
                # Get additional info based on entity type
                if isinstance(entity, types.User):
                    last_active = getattr(entity, 'status', None)
                    if hasattr(last_active, 'was_online'):
                        last_active = last_active.was_online
                
                logger.info(f"❌ {username} is OCCUPIED by {owner_username or owner_id}")
                
                # Update database
                target = TargetUsername(
                    username=username,
                    username_type=username_type,
                    status=status,
                    owner_id=owner_id,
                    owner_username=owner_username,
                    last_checked=datetime.now(),
                    last_active=datetime.now() if username_type == UsernameType.ACCOUNT else None
                )
                
                self.db.update_target(username, **asdict(target))
                self.checked_cache[username] = (False, datetime.now())
                
                return target
                
            except (UsernameNotOccupiedError, ValueError):
                # Username is available
                status = "available"
                logger.info(f"✅ {username} is AVAILABLE!")
                
                target = TargetUsername(
                    username=username,
                    username_type=username_type,
                    status=status,
                    last_checked=datetime.now()
                )
                
                self.db.update_target(username, **asdict(target))
                self.checked_cache[username] = (True, datetime.now())
                
                return target
                
            except Exception as e:
                logger.error(f"Error checking {username}: {e}")
                
                target = TargetUsername(
                    username=username,
                    username_type=username_type,
                    status="error",
                    last_checked=datetime.now()
                )
                
                return target
                
        except FloodWaitError as e:
            logger.warning(f"Flood wait for {username}: {e.seconds}s")
            await asyncio.sleep(e.seconds)
            return await self.check_username(username, username_type)
            
        except Exception as e:
            logger.error(f"Unexpected error checking {username}: {e}")
            raise
    
    async def bulk_check(self, usernames: List[str], username_type: UsernameType = UsernameType.CHANNEL) -> List[TargetUsername]:
        """Check multiple usernames"""
        results = []
        
        # Split into chunks for rate limiting
        chunk_size = 5
        for i in range(0, len(usernames), chunk_size):
            chunk = usernames[i:i + chunk_size]
            
            # Check each username in chunk
            tasks = []
            for username in chunk:
                task = asyncio.create_task(self.check_username(username, username_type))
                tasks.append(task)
            
            # Wait for chunk to complete
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Bulk check error: {result}")
                else:
                    results.append(result)
            
            # Delay between chunks
            if i + chunk_size < len(usernames):
                await asyncio.sleep(2)
        
        return results

# ========== USERNAME ACQUIRER ==========
class UsernameAcquirer:
    """Advanced username acquisition system"""
    
    def __init__(self, client: TelegramClient, database: UsernameDatabase):
        self.client = client
        self.db = database
        self.acquired_count = 0
    
    async def acquire_channel_username(self, username: str) -> AcquisitionResult:
        """Acquire username for a channel"""
        try:
            logger.info(f"🚀 Attempting to acquire {username} for channel...")
            
            # Create channel
            channel_title = f"{Config.CHANNEL_TITLE_PREFIX}{username.title()}"
            
            result = await self.client(CreateChannelRequest(
                title=channel_title,
                about=Config.CHANNEL_ABOUT,
                megagroup=Config.CHANNEL_PRIVATE,
                broadcast=True
            ))
            
            channel = result.chats[0]
            logger.info(f"Channel created: {channel.id}")
            
            # Update username
            await self.client(UpdateUsernameRequest(
                channel=channel,
                username=username
            ))
            
            logger.success(f"🎉 SUCCESS! Channel {username} acquired!")
            
            # Get full channel info
            full_channel = await self.client(GetFullChannelRequest(channel=channel))
            
            result = AcquisitionResult(
                success=True,
                username=username,
                username_type=UsernameType.CHANNEL,
                entity_id=channel.id,
                entity_username=username
            )
            
            self.db.add_acquisition(result)
            self.acquired_count += 1
            
            return result
            
        except UsernameOccupiedError:
            error_msg = f"Username {username} was taken during acquisition"
            logger.error(error_msg)
            
            result = AcquisitionResult(
                success=False,
                username=username,
                username_type=UsernameType.CHANNEL,
                error=error_msg
            )
            
            self.db.add_acquisition(result)
            return result
            
        except FloodWaitError as e:
            error_msg = f"Flood wait: {e.seconds}s"
            logger.warning(error_msg)
            await asyncio.sleep(e.seconds)
            return await self.acquire_channel_username(username)
            
        except Exception as e:
            error_msg = f"Acquisition failed: {e}"
            logger.error(error_msg)
            
            result = AcquisitionResult(
                success=False,
                username=username,
                username_type=UsernameType.CHANNEL,
                error=str(e)
            )
            
            self.db.add_acquisition(result)
            return result
    
    async def acquire_account_username(self, username: str) -> AcquisitionResult:
        """Acquire username for account"""
        try:
            logger.info(f"🚀 Attempting to acquire {username} for account...")
            
            # Update account username
            await self.client(UpdateAccountUsername(
                username=username
            ))
            
            # Update account info
            await self.client(functions.account.UpdateProfileRequest(
                first_name=Config.ACCOUNT_FIRST_NAME,
                last_name=Config.ACCOUNT_LAST_NAME,
                about=Config.ACCOUNT_BIO
            ))
            
            logger.success(f"🎉 SUCCESS! Account {username} acquired!")
            
            result = AcquisitionResult(
                success=True,
                username=username,
                username_type=UsernameType.ACCOUNT,
                entity_username=username
            )
            
            self.db.add_acquisition(result)
            self.acquired_count += 1
            
            return result
            
        except Exception as e:
            error_msg = f"Account acquisition failed: {e}"
            logger.error(error_msg)
            
            result = AcquisitionResult(
                success=False,
                username=username,
                username_type=UsernameType.ACCOUNT,
                error=str(e)
            )
            
            self.db.add_acquisition(result)
            return result
    
    async def quick_acquire(self, target: TargetUsername) -> AcquisitionResult:
        """Quick acquisition attempt"""
        if target.username_type == UsernameType.CHANNEL:
            return await self.acquire_channel_username(target.username)
        elif target.username_type == UsernameType.ACCOUNT:
            return await self.acquire_account_username(target.username)
        else:
            return AcquisitionResult(
                success=False,
                username=target.username,
                username_type=target.username_type,
                error="Unsupported username type"
            )

# ========== ALERT SYSTEM ==========
class AlertSystem:
    """Multi-channel alert system"""
    
    def __init__(self):
        self.last_alert: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(seconds=Config.ALERT_COOLDOWN)
    
    async def send_alert(self, message: str, alert_type: str = "info"):
        """Send alert through configured channels"""
        current_time = datetime.now()
        
        # Check cooldown
        alert_key = f"{alert_type}_{hashlib.md5(message.encode()).hexdigest()[:8]}"
        if alert_key in self.last_alert:
            if current_time - self.last_alert[alert_key] < self.alert_cooldown:
                return
        
        # Update last alert time
        self.last_alert[alert_key] = current_time
        
        # Send through each method
        for method in Config.ALERT_METHODS:
            try:
                if method == 'log':
                    getattr(logger, alert_type)(message)
                    
                elif method == 'file':
                    await self.alert_to_file(message, alert_type)
                    
                elif method == 'telegram' and Config.TELEGRAM_ALERT_CHAT_ID:
                    await self.alert_to_telegram(message, alert_type)
                    
                elif method == 'email':
                    await self.alert_to_email(message, alert_type)
                    
            except Exception as e:
                logger.error(f"Alert method {method} failed: {e}")
    
    async def alert_to_file(self, message: str, alert_type: str):
        """Save alert to file"""
        try:
            alert_file = f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
            async with aiofiles.open(alert_file, 'a') as f:
                await f.write(f"[{datetime.now().isoformat()}] [{alert_type.upper()}] {message}\n")
        except Exception as e:
            logger.error(f"File alert failed: {e}")
    
    async def alert_to_telegram(self, message: str, alert_type: str):
        """Send alert via Telegram"""
        # This would require another TelegramClient instance
        # Implementation depends on your setup
        pass
    
    async def alert_to_email(self, message: str, alert_type: str):
        """Send alert via email"""
        # Email implementation would go here
        pass
    
    async def username_available_alert(self, target: TargetUsername):
        """Alert when username becomes available"""
        message = f"🚨 USERNAME AVAILABLE: {target.username} ({target.username_type.value})"
        await self.send_alert(message, "warning")
    
    async def username_acquired_alert(self, result: AcquisitionResult):
        """Alert when username is acquired"""
        message = f"✅ USERNAME ACQUIRED: {result.username} ({result.username_type.value}) - Entity ID: {result.entity_id}"
        await self.send_alert(message, "success")
    
    async def error_alert(self, error: str, context: str = ""):
        """Alert for errors"""
        message = f"❌ ERROR: {error} {context}"
        await self.send_alert(message, "error")

# ========== MONITORING SYSTEM ==========
class UsernameMonitor:
    """Main monitoring and acquisition system"""
    
    def __init__(self):
        self.client: Optional[TelegramClient] = None
        self.db = UsernameDatabase()
        self.checker: Optional[UsernameChecker] = None
        self.acquirer: Optional[UsernameAcquirer] = None
        self.alerter = AlertSystem()
        self.running = False
        self.session_file = f"{Config.SESSION_DIR}/hunter_session"
        
        # Ensure directories exist
        Path(Config.SESSION_DIR).mkdir(exist_ok=True)
        Path(Config.BACKUP_DIR).mkdir(exist_ok=True)
    
    async def initialize_client(self):
        """Initialize Telegram client"""
        try:
            # Create client with advanced settings
            self.client = TelegramClient(
                session=self.session_file,
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                device_model="Username Hunter Pro",
                system_version="HunterOS 1.0",
                app_version="10.0.0",
                lang_code="en",
                system_lang_code="en-US",
                connection_retries=5,
                request_retries=5,
                flood_sleep_threshold=60
            )
            
            # Connect and authenticate
            await self.client.connect()
            
            if not await self.client.is_user_authorized():
                logger.info("Authenticating...")
                await self.client.start(phone=Config.PHONE_NUMBER)
            
            # Get current user info
            me = await self.client.get_me()
            logger.info(f"Authenticated as: @{me.username} ({me.id})")
            
            # Initialize subsystems
            self.checker = UsernameChecker(self.client, self.db)
            self.acquirer = UsernameAcquirer(self.client, self.db)
            
            return True
            
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            return False
    
    async def load_targets(self):
        """Load and initialize targets"""
        targets_added = 0
        
        for username in Config.TARGET_USERNAMES:
            for username_type_str in Config.USERNAME_TYPES:
                username_type = UsernameType(username_type_str)
                
                # Check if target already exists
                target_key = f"{username}_{username_type.value}"
                if target_key not in self.db.targets:
                    # Create new target
                    priority = 2 if username in Config.PRIORITY_USERNAMES else 1
                    
                    target = TargetUsername(
                        username=username,
                        username_type=username_type,
                        priority=priority,
                        last_checked=None,
                        status="unknown"
                    )
                    
                    self.db.add_target(target)
                    targets_added += 1
        
        if targets_added:
            logger.info(f"Added {targets_added} new targets to database")
    
    async def check_all_targets(self):
        """Check all targets"""
        logger.info("Starting comprehensive target check...")
        
        # Group targets by priority
        high_priority = [t for t in self.db.targets.values() if t.priority >= 2]
        normal_priority = [t for t in self.db.targets.values() if t.priority == 1]
        
        # Check high priority targets first
        if high_priority:
            logger.info(f"Checking {len(high_priority)} high priority targets")
            await self.check_target_group(high_priority)
        
        # Check normal priority targets
        if normal_priority:
            logger.info(f"Checking {len(normal_priority)} normal priority targets")
            await self.check_target_group(normal_priority)
        
        # Print stats
        stats = self.db.get_stats()
        logger.info(f"Stats: {stats['available']} available, {stats['occupied']} occupied, {stats['unknown']} unknown")
    
    async def check_target_group(self, targets: List[TargetUsername]):
        """Check group of targets"""
        for target in targets:
            try:
                # Check username
                result = await self.checker.check_username(target.username, target.username_type)
                
                # If available, try to acquire immediately
                if result.is_available():
                    logger.warning(f"🚨 {target.username} is AVAILABLE! Attempting acquisition...")
                    await self.alerter.username_available_alert(result)
                    
                    # Attempt acquisition
                    acquisition_result = await self.acquirer.quick_acquire(result)
                    
                    if acquisition_result.success:
                        await self.alerter.username_acquired_alert(acquisition_result)
                        logger.success(f"Successfully acquired {target.username}")
                    else:
                        logger.error(f"Failed to acquire {target.username}: {acquisition_result.error}")
                
                # Small delay between checks
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error checking {target.username}: {e}")
                await self.alerter.error_alert(str(e), f"while checking {target.username}")
    
    async def continuous_monitoring(self):
        """Continuous monitoring loop"""
        logger.info("Starting continuous monitoring...")
        self.running = True
        
        iteration = 0
        while self.running:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Monitoring iteration #{iteration}")
            logger.info(f"{'='*60}")
            
            try:
                # Check all targets
                await self.check_all_targets()
                
                # Print summary
                self.print_summary()
                
                # Save database
                self.db.save_data()
                
                # Calculate next check interval
                next_check = Config.QUICK_CHECK_INTERVAL
                logger.info(f"Next check in {next_check} seconds...")
                
                # Wait for next check
                for _ in range(next_check):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(Config.RETRY_DELAY)
    
    def print_summary(self):
        """Print monitoring summary"""
        stats = self.db.get_stats()
        
        print("\n" + "="*60)
        print("USERNAME HUNTER - STATUS SUMMARY")
        print("="*60)
        print(f"Total targets: {stats['total_targets']}")
        print(f"Available: {stats['available']} | Occupied: {stats['occupied']}")
        print(f"Acquisitions: {stats['total_acquisitions']}")
        print(f"Total attempts: {stats['total_attempts']}")
        
        # Show recently checked targets
        recent_targets = sorted(
            self.db.targets.values(),
            key=lambda x: x.last_checked or datetime.min,
            reverse=True
        )[:5]
        
        if recent_targets:
            print("\nRecently checked:")
            for target in recent_targets:
                status_icon = "✅" if target.status == "available" else "❌" if target.status == "occupied" else "❓"
                print(f"  {status_icon} {target.username} ({target.username_type.value}) - {target.last_checked.strftime('%H:%M:%S') if target.last_checked else 'Never'}")
    
    async def quick_monitor(self, username: str):
        """Quick monitor for specific username"""
        logger.info(f"Quick monitoring: {username}")
        
        target = TargetUsername(
            username=username,
            username_type=UsernameType.CHANNEL,
            priority=3
        )
        
        check_count = 0
        while self.running and check_count < 100:  # Max 100 quick checks
            check_count += 1
            
            result = await self.checker.check_username(target.username, target.username_type)
            
            if result.is_available():
                logger.warning(f"🚨 {username} became available!")
                await self.alerter.username_available_alert(result)
                
                # Try to acquire
                acquisition_result = await self.acquirer.quick_acquire(result)
                if acquisition_result.success:
                    logger.success(f"Acquired {username}!")
                    break
            
            # Quick delay
            await asyncio.sleep(1)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        self.running = False
        
        if self.client:
            await self.client.disconnect()
        
        self.db.save_data()
        logger.info("Shutdown complete")

# ========== COMMAND LINE INTERFACE ==========
async def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      PROFESSIONAL TELEGRAM USERNAME HUNTER              ║
    ║                 HEVORIX EDITION v3.0                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Telegram Username Hunter')
    parser.add_argument('--mode', choices=['monitor', 'quick', 'check', 'stats'], 
                       default='monitor', help='Operation mode')
    parser.add_argument('--username', type=str, help='Specific username to target')
    parser.add_argument('--list', action='store_true', help='List all targets')
    parser.add_argument('--add', type=str, help='Add new username to monitor')
    parser.add_argument('--remove', type=str, help='Remove username from monitoring')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3], 
                       help='Set priority for username (1=low, 3=high)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = UsernameMonitor()
    
    try:
        # Initialize client
        if not await monitor.initialize_client():
            logger.error("Failed to initialize Telegram client")
            return
        
        # Load targets
        await monitor.load_targets()
        
        # Handle different modes
        if args.mode == 'monitor':
            # Continuous monitoring
            await monitor.continuous_monitoring()
            
        elif args.mode == 'quick' and args.username:
            # Quick monitor specific username
            await monitor.quick_monitor(args.username)
            
        elif args.mode == 'check':
            # Single check of all targets
            await monitor.check_all_targets()
            monitor.print_summary()
            
        elif args.mode == 'stats':
            # Show statistics
            stats = monitor.db.get_stats()
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Show targets
            if args.list:
                print("\nTargets:")
                for target in monitor.db.targets.values():
                    print(f"  {target.username} ({target.username_type.value}) - {target.status}")
        
        elif args.add:
            # Add new target
            username_type = UsernameType.CHANNEL
            target = TargetUsername(
                username=args.add,
                username_type=username_type,
                priority=args.priority or 1
            )
            monitor.db.add_target(target)
            logger.info(f"Added {args.add} to monitoring")
            
        elif args.remove:
            # Remove target
            if args.remove in monitor.db.targets:
                del monitor.db.targets[args.remove]
                monitor.db.save_data()
                logger.info(f"Removed {args.remove} from monitoring")
            else:
                logger.error(f"Target {args.remove} not found")
        
        else:
            # Default: continuous monitoring
            await monitor.continuous_monitoring()
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        await monitor.shutdown()

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Ensure asyncio event loop is properly handled
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
