import time
import pygame
import pymunk
import pymunk.pygame_util
from youtube import get_live_stream, get_new_live_chat_messages, get_live_chat_id, get_subscriber_count, validate_live_stream_id
from config import config
from donation import classify as classify_donation
from atlas import create_texture_atlas
from pathlib import Path
from chunk import get_block, clean_chunks, delete_block, chunks
from constants import BLOCK_SCALE_FACTOR, BLOCK_SIZE, CHUNK_HEIGHT, CHUNK_WIDTH, INTERNAL_HEIGHT, INTERNAL_WIDTH, FRAMERATE
from pickaxe import Pickaxe
from camera import Camera
from sound import SoundManager
from tnt import Tnt, MegaTnt
from explosion import Explosion
from goldrain import GoldShower, ConfettiShower
import asyncio
import threading
import random
from hud import Hud
from collections import deque

# Track key states
key_t_pressed = False
key_m_pressed = False

#
live_stream = None
live_chat_id = None
subscribers = None

if config["CHAT_CONTROL"] == True:
    print("Checking for specific live stream")
    if config["LIVESTREAM_ID"] is not None and config["LIVESTREAM_ID"] != "":
        stream_id = validate_live_stream_id(config["LIVESTREAM_ID"])
        if stream_id is not None:
            live_stream = get_live_stream(stream_id)

    if live_stream is None:
        print("No specific live stream found. App will run without it.")
    else:
        print("Live stream found:", live_stream["snippet"]["title"])

    # get chat id from live stream
    if live_stream is not None:
        print("Fetching live chat ID...")
        live_chat_id = get_live_chat_id(live_stream["id"])

    if live_chat_id is None:
        print("No live chat ID found. App will run without it.")
    else:
        print("Live chat ID found:", live_chat_id)

    # get subscribers count
    if(config["CHANNEL_ID"] is not None and config["CHANNEL_ID"] != ""):
        print("Fetching subscribers count...")
        subscribers = get_subscriber_count(config["CHANNEL_ID"])

    if subscribers is None:
        print("No subscribers count found. App will run without it.")
    else:
        print("Subscribers count found:", subscribers)

# Queues for chat
tnt_queue = deque()
tnt_queue_authors = set()
tnt_superchat_queue = deque()
tnt_superchat_authors = set()
fast_slow_queue = deque()
fast_slow_authors = set()
big_queue = deque()
big_authors = set()
pickaxe_queue = deque()
pickaxe_authors = set()
mega_tnt_queue = deque()

# 재미 커맨드 큐 (nuke / missile / meteor / earthquake / gold rain)
nuke_queue = deque()
nuke_authors = set()
missile_queue = deque()
missile_authors = set()
meteor_queue = deque()
meteor_authors = set()
earthquake_queue = deque()
earthquake_authors = set()
goldrain_queue = deque()
goldrain_authors = set()
freeze_queue = deque()
freeze_authors = set()
tiny_queue = deque()
tiny_authors = set()
confetti_queue = deque()
confetti_authors = set()
lucky_queue = deque()
lucky_authors = set()

# 재미 커맨드 5종의 실제 실행 로직. 채팅 명령과 45초 자동 랜덤 이벤트 양쪽에서 재사용한다.
def trigger_nuke(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, camera, hud, author):
    new_nuke = MegaTnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                        texture_atlas, atlas_items, sound_manager, owner_name=author, scale_multiplier=6)
    tnt_list.append(new_nuke)
    camera.shake(45, 40)
    hud.add_screen_flash((255, 255, 255), 300, peak_alpha=220)
    hud.add_event_banner(f"{author}'s NUKE!", (231, 76, 60), 3500)
    hud.add_command_feed(author, "NUKE")


def trigger_missile(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, author):
    for _ in range(6):
        offset_x = random.randint(-220, 220)
        offset_y = random.randint(-250, 50)
        spawn_pos = (pickaxe.body.position.x + offset_x, pickaxe.body.position.y - 150 + offset_y)
        new_missile = Tnt(space, spawn_pos[0], spawn_pos[1],
                           texture_atlas, atlas_items, sound_manager, owner_name=author)
        tnt_list.append(new_missile)
        # 발사 순간 작은 연기 파티클을 남겨서 미사일이 날아온 느낌을 준다
        explosions.append(Explosion(spawn_pos, texture_atlas, atlas_items, particle_count=5))
    camera.shake(20, 18)
    hud.add_event_banner(f"{author}'s MISSILE BARRAGE!", (230, 126, 34), 3000)
    hud.add_command_feed(author, "MISSILE")


def trigger_meteor(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, author):
    new_meteor = MegaTnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                          texture_atlas, atlas_items, sound_manager, owner_name=author, scale_multiplier=3)
    tnt_list.append(new_meteor)
    for i in range(4):
        trail_pos = pickaxe.body.position + (0, -100 - i * 80)
        explosions.append(Explosion(trail_pos, texture_atlas, atlas_items, particle_count=8))
    camera.shake(25, 20)
    hud.add_screen_flash((255, 140, 60), 200, peak_alpha=140)
    hud.add_event_banner(f"{author}'s METEOR!", (192, 57, 43), 3000)
    hud.add_command_feed(author, "METEOR")


def trigger_earthquake(camera, hud, author):
    camera.shake(50, 30)
    hud.add_event_banner(f"{author} caused an EARTHQUAKE!", (142, 68, 173), 2500)
    hud.add_command_feed(author, "EARTHQUAKE")


def trigger_gold_rain(pickaxe, texture_atlas, atlas_items, shower_list, hud, author):
    shower_list.append(GoldShower(pickaxe.body.position.y - 100, texture_atlas, atlas_items))
    hud.add_event_banner(f"{author}'s GOLD RAIN!", (241, 196, 15), 3000)
    hud.add_command_feed(author, "GOLD RAIN")


def trigger_freeze(pickaxe, hud, author):
    pickaxe.freeze(2000)
    hud.add_event_banner(f"{author} froze the pickaxe!", (120, 190, 255), 2000)
    hud.add_command_feed(author, "FREEZE")


def trigger_tiny(pickaxe, hud, author):
    pickaxe.shrink(4000)
    hud.add_command_feed(author, "TINY")


def trigger_confetti(pickaxe, shower_list, hud, author):
    shower_list.append(ConfettiShower(pickaxe.body.position.y - 100))
    hud.add_event_banner(f"{author}'s CONFETTI!", (230, 126, 194), 2500)
    hud.add_command_feed(author, "CONFETTI")


def trigger_lucky(hud, author):
    bonus = {
        "coal": random.randint(1, 3),
        "iron_ingot": random.randint(0, 2),
        "gold_ingot": random.randint(0, 2),
        "diamond": random.randint(0, 1),
    }
    hud.update_amounts({name: hud.amounts[name] + amount for name, amount in bonus.items()})
    hud.add_event_banner(f"{author} got LUCKY!", (46, 204, 113), 2200)
    hud.add_command_feed(author, "LUCKY")


async def handle_youtube_poll():
    global subscribers # Use global to modify the variable

    if subscribers is not None:
        new_subscribers = get_subscriber_count(config["CHANNEL_ID"])
        if new_subscribers is not None and new_subscribers > subscribers:
            mega_tnt_queue.append("New Subscriber") # Add to mega tnt queue
            subscribers = new_subscribers # Update subscriber count

    new_messages = get_new_live_chat_messages(live_chat_id)

    for message in new_messages:
        author = message["author"]
        text = message["message"]
        is_superchat = message["sc_details"] is not None
        is_supersticker = message["ss_details"] is not None

        text_lower = text.lower()

        # Check for "tnt" command (add author to regular tnt_queue) - Only English "tnt"
        if "tnt" in text_lower:
            if author not in tnt_queue_authors:
                tnt_queue.append(author)
                tnt_queue_authors.add(author)
                print(f"Added {author} to regular TNT queue")

        # Check for Superchat/Supersticker (add to superchat tnt queue, classified by donation tier)
        if is_superchat or is_supersticker:
            if author not in tnt_superchat_authors:
                 details = message["sc_details"] or message["ss_details"]
                 tier_info = classify_donation(details)
                 amount_text = details.get("amountDisplayString", "") if details else ""
                 tnt_superchat_queue.append((author, text, tier_info, amount_text))
                 tnt_superchat_authors.add(author)
                 print(f"Added {author} to Superchat TNT queue (tier {tier_info['tier']}, {amount_text})")

        if "fast" in text.lower() and author not in fast_slow_authors:
            fast_slow_queue.append((author, "Fast"))
            fast_slow_authors.add(author)
            print(f"Added {author} to Fast/Slow queue (Fast)")
        elif "slowmo" in text.lower() and author not in fast_slow_authors:
            # "slowmo"는 "slow"의 부분 문자열이라 slow보다 먼저 검사해야 한다
            fast_slow_queue.append((author, "SlowMo"))
            fast_slow_authors.add(author)
            print(f"Added {author} to Fast/Slow queue (SlowMo)")
        elif "slow" in text.lower() and author not in fast_slow_authors:
            fast_slow_queue.append((author, "Slow"))
            fast_slow_authors.add(author)
            print(f"Added {author} to Fast/Slow queue (Slow)")

        if "big" in text.lower() and author not in big_authors:
            big_queue.append(author)
            big_authors.add(author)
            print(f"Added {author} to Big queue")

        # Check for pickaxe commands (add author and pickaxe type to pickaxe_queue)
        if "wood" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "wooden_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (wooden_pickaxe)")
        elif "stone" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "stone_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (stone_pickaxe)")
        elif "iron" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "iron_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (iron_pickaxe)")
        elif "gold" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "golden_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (golden_pickaxe)")
        elif "diamond" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "diamond_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (diamond_pickaxe)")
        elif "netherite" in text_lower:
             if author not in pickaxe_authors:
                 pickaxe_queue.append((author, "netherite_pickaxe"))
                 pickaxe_authors.add(author)
                 print(f"Added {author} to Pickaxe queue (netherite_pickaxe)")

        # 재미 커맨드: nuke / missile / meteor / earthquake / gold rain(shower)
        if "nuke" in text_lower and author not in nuke_authors:
            nuke_queue.append(author)
            nuke_authors.add(author)
            print(f"Added {author} to Nuke queue")

        if "missile" in text_lower and author not in missile_authors:
            missile_queue.append(author)
            missile_authors.add(author)
            print(f"Added {author} to Missile queue")

        if "meteor" in text_lower and author not in meteor_authors:
            meteor_queue.append(author)
            meteor_authors.add(author)
            print(f"Added {author} to Meteor queue")

        if "earthquake" in text_lower and author not in earthquake_authors:
            earthquake_queue.append(author)
            earthquake_authors.add(author)
            print(f"Added {author} to Earthquake queue")

        if "shower" in text_lower and author not in goldrain_authors:
            goldrain_queue.append(author)
            goldrain_authors.add(author)
            print(f"Added {author} to Gold rain queue")

        if "freeze" in text_lower and author not in freeze_authors:
            freeze_queue.append(author)
            freeze_authors.add(author)
            print(f"Added {author} to Freeze queue")

        if "tiny" in text_lower and author not in tiny_authors:
            tiny_queue.append(author)
            tiny_authors.add(author)
            print(f"Added {author} to Tiny queue")

        if "confetti" in text_lower and author not in confetti_authors:
            confetti_queue.append(author)
            confetti_authors.add(author)
            print(f"Added {author} to Confetti queue")

        if "lucky" in text_lower and author not in lucky_authors:
            lucky_queue.append(author)
            lucky_authors.add(author)
            print(f"Added {author} to Lucky queue")

    # print the queue counts (optional, for debugging)
    # print(f"Queues: TNT={len(tnt_queue)}, Superchat TNT={len(tnt_superchat_queue)}, Fast/Slow={len(fast_slow_queue)}, Big={len(big_queue)}, Pickaxe={len(pickaxe_queue)}, MegaTNT={len(mega_tnt_queue)}")

def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Create a new event loop
asyncio_loop = asyncio.new_event_loop()
# Start it in a daemon thread so it doesn’t block shutdown
threading.Thread(target=start_event_loop, args=(asyncio_loop,), daemon=True).start()

def game():
    # Initialize pygame
    pygame.init()
    clock = pygame.time.Clock()

    # 창은 가능하면 내부 렌더 해상도(1080x1920)와 동일하게 띄워서 화질 손실을 피하지만,
    # 노트북처럼 화면이 그보다 작으면 화면 높이의 85%를 넘지 않도록 자동으로 줄인다.
    # (내부 렌더링 자체는 항상 1080x1920 그대로라, 창이 작아도 smoothscale로 부드럽게 축소될 뿐 화질이 뭉개지진 않는다)
    display_info = pygame.display.Info()
    max_window_height = int(display_info.current_h * 0.85)
    window_height = min(INTERNAL_HEIGHT, max_window_height)
    window_width = int(window_height * (INTERNAL_WIDTH / INTERNAL_HEIGHT))

    # Pymunk physics
    space = pymunk.Space()
    space.gravity = (0, 1000)  # (x, y) - down is positive y

    # Create a resizable window
    screen_size = (window_width, window_height)
    screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
    scaled_surface = pygame.Surface(screen_size).convert()
    pygame.display.set_caption("Falling Pickaxe")
    # set icon
    icon = pygame.image.load(Path(__file__).parent.parent / "src/assets/pickaxe" / "diamond_pickaxe.png")
    pygame.display.set_icon(icon)

    # Create an internal surface with fixed resolution
    internal_surface = pygame.Surface((INTERNAL_WIDTH, INTERNAL_HEIGHT))

    # Load texture atlas
    assets_dir = Path(__file__).parent.parent / "src/assets"
    (texture_atlas, atlas_items) = create_texture_atlas(assets_dir)

    # Load background
    background_image = pygame.image.load(assets_dir / "background.png")
    background_scale_factor = 1.5
    background_width = int(background_image.get_width() * background_scale_factor)
    background_height = int(background_image.get_height() * background_scale_factor)
    background_image = pygame.transform.scale(background_image, (background_width, background_height))

    # Scale the entire texture atlas
    texture_atlas = pygame.transform.scale(texture_atlas,
                                        (texture_atlas.get_width() * BLOCK_SCALE_FACTOR,
                                        texture_atlas.get_height() * BLOCK_SCALE_FACTOR))

    for category in atlas_items:
        for item in atlas_items[category]:
            x, y, w, h = atlas_items[category][item]
            atlas_items[category][item] = (x * BLOCK_SCALE_FACTOR, y * BLOCK_SCALE_FACTOR, w * BLOCK_SCALE_FACTOR, h * BLOCK_SCALE_FACTOR)

    #sounds
    sound_manager = SoundManager()

    sound_manager.load_sound("tnt", assets_dir / "sounds" / "tnt.mp3", 0.3)
    sound_manager.load_sound("stone1", assets_dir / "sounds" / "stone1.wav", 0.5)
    sound_manager.load_sound("stone2", assets_dir / "sounds" / "stone2.wav", 0.5)
    sound_manager.load_sound("stone3", assets_dir / "sounds" / "stone3.wav", 0.5)
    sound_manager.load_sound("stone4", assets_dir / "sounds" / "stone4.wav", 0.5)
    sound_manager.load_sound("grass1", assets_dir / "sounds" / "grass1.wav", 0.1)
    sound_manager.load_sound("grass2", assets_dir / "sounds" / "grass2.wav", 0.1)
    sound_manager.load_sound("grass3", assets_dir / "sounds" / "grass3.wav", 0.1)
    sound_manager.load_sound("grass4", assets_dir / "sounds" / "grass4.wav", 0.1)

    # Pickaxe
    pickaxe = Pickaxe(space, INTERNAL_WIDTH // 2, INTERNAL_HEIGHT // 2, texture_atlas.subsurface(atlas_items["pickaxe"]["wooden_pickaxe"]), sound_manager)

    # TNT
    tnt_list = []  # List to keep track of spawned TNT objects

    # Pickaxe enlargement
    enlarge_duration = 1000 * config["PICKAXE_ENLARGE_DURATION_SECONDS"]

    # Fast slow
    fast_slow_active = False
    fast_slow = random.choice(["Fast", "Slow"])
    last_fast_slow = pygame.time.get_ticks()

    # Camera
    camera = Camera()

    # HUD
    hud = Hud(texture_atlas, atlas_items)

    # Explosions
    explosions = []

    # 골드비/색종이 등 물리 충돌 없는 장식용 낙하 파티클 (GoldShower, ConfettiShower가 같은 인터페이스를 공유)
    decorative_showers = []

    # Youtube
    yt_poll_interval = 1000 * config["YT_POLL_INTERVAL_SECONDS"]
    last_yt_poll = pygame.time.get_ticks()

    # Save progress interval
    save_progress_interval = 1000 * config["SAVE_PROGRESS_INTERVAL_SECONDS"]
    last_save_progress = pygame.time.get_ticks()

    # Youtupe chat queues
    queues_pop_interval = 1000 * config["QUEUES_POP_INTERVAL_SECONDS"]
    last_queues_pop = pygame.time.get_ticks()

    # 45초마다 자동으로 재미 랜덤 이벤트(nuke/missile/meteor/earthquake/gold rain) 발생
    random_event_interval = 45 * 1000
    last_random_event = pygame.time.get_ticks()

    # 2초마다 콘솔에 FPS 출력 (화면에는 안 띄움 - 방송 화면을 가리지 않기 위해 터미널에서만 확인)
    last_fps_print = pygame.time.get_ticks()

    # 콤보: 일정 시간 안에 연속으로 블록을 부수면 콤보가 쌓이고, 특정 수치를 넘으면 보상 이벤트가 자동으로 터진다
    combo_count = 0
    best_combo = 0
    last_break_time = pygame.time.get_ticks()
    combo_timeout_ms = 2500
    combo_milestones_hit = set()

    # Main loop
    running = True
    user_quit = False
    while running:
        # ++++++++++++++++++  EVENTS ++++++++++++++++++
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Close window event
                running = False
                user_quit = True
            elif event.type == pygame.VIDEORESIZE:  # Window resize event
                new_width, new_height = event.w, event.h

                # Maintain 9:16 aspect ratio
                if new_width / 9 > new_height / 16:
                    new_width = int(new_height * (9 / 16))
                else:
                    new_height = int(new_width * (16 / 9))

                window_width, window_height = new_width, new_height
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                scaled_surface = pygame.Surface((window_width, window_height)).convert()

        # ++++++++++++++++++  UPDATE ++++++++++++++++++
        # Determine which chunks are visible
        # Update physics

        dt_ms = clock.get_time()
        current_time = pygame.time.get_ticks()

        step_speed = 1 / FRAMERATE  # Fixed time step for physics simulation
        if fast_slow_active and fast_slow == "Fast":
            step_speed = 1 / (FRAMERATE / 2)
        elif fast_slow_active and fast_slow == "Slow":
            step_speed = 1 / (FRAMERATE * 2)
        elif fast_slow_active and fast_slow == "SlowMo":
            step_speed = 1 / (FRAMERATE * 5)

        space.step(step_speed)

        start_chunk_y = int(pickaxe.body.position.y // (CHUNK_HEIGHT * BLOCK_SIZE) - 1) - 1
        end_chunk_y = int(pickaxe.body.position.y + INTERNAL_HEIGHT) // (CHUNK_HEIGHT * BLOCK_SIZE)  + 1

        # Update pickaxe
        pickaxe.update(current_time)

        # Update camera
        camera.update(pickaxe.body.position.y)

        # ++++++++++++++++++  DRAWING ++++++++++++++++++
        # Clear the internal surface
        screen.fill((0, 0, 0))

        # Fill internal surface with the background
        internal_surface.blit(background_image, ((INTERNAL_WIDTH - background_width) // 2, (INTERNAL_HEIGHT - background_height) // 2))

        # TNT/곡괭이 변경/확대/속도는 더 이상 자체적으로 랜덤 타이머를 돌리지 않는다.
        # 이제 자동 연출은 45초 랜덤 이벤트 하나로 통일되고, 그 외에는 채팅 명령/키 입력에만 반응한다.
        # (진행 중인 fast/slow 효과는 지속 시간이 끝나면 정상적으로 종료시켜야 한다)
        if fast_slow_active and current_time - last_fast_slow >= (1000 * config["FAST_SLOW_DURATION_SECONDS"]):
            fast_slow_active = False
            last_fast_slow = current_time

        # 이 프레임 안에서 자동 이벤트(45초 타이머든 콤보 마일스톤이든)가 이미 하나 터졌으면
        # 같은 프레임에 또 터지지 않도록 막는 공용 플래그. 두 시스템이 각자 조건을 체크하다
        # 우연히 같은 프레임에 동시에 참이 되면서 이벤트가 겹쳐 나오던 버그를 막는다.
        spectacle_fired_this_frame = False

        # Check if it's time for the automatic 45s random spectacle event
        if current_time - last_random_event >= random_event_interval:
            last_random_event = current_time
            spectacle_fired_this_frame = True
            event_choice = random.choice(["nuke", "missile", "meteor", "earthquake", "goldrain"])
            if event_choice == "nuke":
                trigger_nuke(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, camera, hud, "Random Event")
            elif event_choice == "missile":
                trigger_missile(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, "Random Event")
            elif event_choice == "meteor":
                trigger_meteor(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, "Random Event")
            elif event_choice == "earthquake":
                trigger_earthquake(camera, hud, "Random Event")
            else:
                trigger_gold_rain(pickaxe, texture_atlas, atlas_items, decorative_showers, hud, "Random Event")

        # Update all TNTs
        for tnt in tnt_list:
            tnt.update(tnt_list, explosions, camera, current_time)

        # Poll Yotutube api
        if live_chat_id is not None and current_time - last_yt_poll >= yt_poll_interval:
            print("Polling YouTube API...")
            last_yt_poll = current_time
            asyncio.run_coroutine_threadsafe(handle_youtube_poll(), asyncio_loop)

        # Process chat queues
        if config["CHAT_CONTROL"] and current_time - last_queues_pop >= queues_pop_interval:
            last_queues_pop = current_time

            # Handle regular TNT from chat command
            if tnt_queue:
                author = tnt_queue.popleft()
                tnt_queue_authors.discard(author)
                print(f"Spawning regular TNT for {author} (from chat command)")
                new_tnt = Tnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                             texture_atlas, atlas_items, sound_manager, owner_name=author)
                tnt_list.append(new_tnt)
                hud.add_command_feed(author, "TNT")

            # Handle MegaTNT (New Subscriber)
            if mega_tnt_queue:
                author = mega_tnt_queue.popleft()
                print(f"Spawning MegaTNT for {author} (New Subscriber)")
                new_megatnt = MegaTnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                      texture_atlas, atlas_items, sound_manager, owner_name=author)
                tnt_list.append(new_megatnt)
                hud.add_command_feed(author, "신규 구독 MegaTNT")

            # Handle Superchat/Supersticker TNT, scaled by donation tier
            if tnt_superchat_queue:
                author, text, tier_info, amount_text = tnt_superchat_queue.popleft()
                tnt_superchat_authors.discard(author)
                print(f"Spawning TNT for {author} (Superchat tier {tier_info['tier']}: {text})")
                for _ in range(tier_info["tnt_count"]):
                    if tier_info.get("mega"):
                        new_tnt = MegaTnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                                           texture_atlas, atlas_items, sound_manager, owner_name=author,
                                           scale_multiplier=tier_info.get("mega_scale", 2))
                    else:
                        new_tnt = Tnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                                       texture_atlas, atlas_items, sound_manager, owner_name=author)
                    tnt_list.append(new_tnt)
                camera.shake(tier_info["shake_duration"], tier_info["shake_intensity"])
                if tier_info["tier"] >= 4:
                    hud.add_screen_flash((255, 215, 80), 280, peak_alpha=200)
                hud.add_donation_alert(author, amount_text or "후원", tier_info["tier"])
                hud.add_command_feed(author, f"후원 {amount_text} (티어 {tier_info['tier']})")

            # Handle Fast/Slow command
            if fast_slow_queue:
                author, q_fast_slow = fast_slow_queue.popleft()
                fast_slow_authors.discard(author)
                print(f"Changing speed for {author} to {q_fast_slow}")
                fast_slow_active = True
                last_fast_slow = current_time
                fast_slow = q_fast_slow
                hud.add_command_feed(author, q_fast_slow)

            # Handle Big pickaxe command
            if big_queue:
                author = big_queue.popleft()
                big_authors.discard(author)
                print(f"Making pickaxe big for {author}")
                pickaxe.enlarge(enlarge_duration)
                hud.add_command_feed(author, "BIG")

            # Handle Pickaxe type command
            if pickaxe_queue:
                author, pickaxe_type = pickaxe_queue.popleft()
                pickaxe_authors.discard(author)
                print(f"Changing pickaxe for {author} to {pickaxe_type}")
                pickaxe.pickaxe(pickaxe_type, texture_atlas, atlas_items)
                hud.add_command_feed(author, pickaxe_type)

            # Handle Nuke command (초대형 MegaTNT)
            if nuke_queue:
                author = nuke_queue.popleft()
                nuke_authors.discard(author)
                print(f"Spawning NUKE for {author}")
                trigger_nuke(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, camera, hud, author)

            # Handle Missile command (여러 발 동시 투하)
            if missile_queue:
                author = missile_queue.popleft()
                missile_authors.discard(author)
                print(f"Spawning missile barrage for {author}")
                trigger_missile(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, author)

            # Handle Meteor command (거대 불덩이 + 화염 트레일)
            if meteor_queue:
                author = meteor_queue.popleft()
                meteor_authors.discard(author)
                print(f"Spawning meteor for {author}")
                trigger_meteor(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, author)

            # Handle Earthquake command (폭발 없이 화면 흔들림만)
            if earthquake_queue:
                author = earthquake_queue.popleft()
                earthquake_authors.discard(author)
                print(f"Triggering earthquake for {author}")
                trigger_earthquake(camera, hud, author)

            # Handle Gold rain command (장식용 낙하 파티클)
            if goldrain_queue:
                author = goldrain_queue.popleft()
                goldrain_authors.discard(author)
                print(f"Spawning gold rain for {author}")
                trigger_gold_rain(pickaxe, texture_atlas, atlas_items, decorative_showers, hud, author)

            # Handle Freeze command (잠깐 중력 무시하고 정지)
            if freeze_queue:
                author = freeze_queue.popleft()
                freeze_authors.discard(author)
                print(f"Freezing pickaxe for {author}")
                trigger_freeze(pickaxe, hud, author)

            # Handle Tiny command (big의 반대, 곡괭이 축소)
            if tiny_queue:
                author = tiny_queue.popleft()
                tiny_authors.discard(author)
                print(f"Shrinking pickaxe for {author}")
                trigger_tiny(pickaxe, hud, author)

            # Handle Confetti command (컬러풀한 장식용 낙하 파티클)
            if confetti_queue:
                author = confetti_queue.popleft()
                confetti_authors.discard(author)
                print(f"Spawning confetti for {author}")
                trigger_confetti(pickaxe, decorative_showers, hud, author)

            # Handle Lucky command (즉시 소량의 랜덤 광물 보너스)
            if lucky_queue:
                author = lucky_queue.popleft()
                lucky_authors.discard(author)
                print(f"Rolling lucky bonus for {author}")
                trigger_lucky(hud, author)


        # Delete chunks
        clean_chunks(start_chunk_y, space)

        # Draw blocks in visible chunks
        for chunk_x in range(-1, 2):
            for chunk_y in range(start_chunk_y, end_chunk_y):
                for y in range(CHUNK_HEIGHT):
                    for x in range(CHUNK_WIDTH):
                        block = get_block(chunk_x, chunk_y, x, y, texture_atlas, atlas_items, space)

                        if block == None:
                            continue

                        just_destroyed = block.update(space, hud, current_time)
                        block.draw(internal_surface, camera)

                        if just_destroyed:
                            combo_count += 1
                            last_break_time = current_time
                            if combo_count > best_combo:
                                best_combo = combo_count
                                hud.add_event_banner("NEW BEST COMBO!", (255, 215, 80), 2000)

        # 콤보가 일정 시간 갱신되지 않으면 초기화
        if combo_count > 0 and current_time - last_break_time > combo_timeout_ms:
            combo_count = 0
            combo_milestones_hit.clear()

        # 콤보 마일스톤 보상: 정해진 콤보 수치를 넘을 때마다 자동으로 화려한 이벤트가 터진다.
        # 45초 자동 랜덤 이벤트 타이머도 여기서 같이 리셋해서, 다음 프레임부터는 두 시스템이 안 겹치게 한다.
        # 그리고 이 프레임에 이미 다른 자동 이벤트가 터졌으면(spectacle_fired_this_frame) 밀스톤은
        # combo_milestones_hit에 추가하지 않고 넘어가서, 바로 다음 프레임에 혼자 터지도록 미룬다.
        if not spectacle_fired_this_frame:
            for milestone in (10, 25, 50, 100):
                if combo_count >= milestone and milestone not in combo_milestones_hit:
                    combo_milestones_hit.add(milestone)
                    last_random_event = current_time
                    spectacle_fired_this_frame = True
                    label = f"COMBO x{milestone}!"
                    if milestone == 10:
                        trigger_gold_rain(pickaxe, texture_atlas, atlas_items, decorative_showers, hud, label)
                    elif milestone == 25:
                        trigger_missile(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, label)
                    elif milestone == 50:
                        trigger_meteor(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, explosions, camera, hud, label)
                    elif milestone == 100:
                        trigger_nuke(space, pickaxe, texture_atlas, atlas_items, sound_manager, tnt_list, camera, hud, label)
                    break

        hud.set_combo(combo_count, best_combo)

        # Draw pickaxe
        pickaxe.draw(internal_surface, camera)

        # Draw TNT
        for tnt in tnt_list:
            tnt.draw(internal_surface, camera)

        # Draw particles
        for explosion in explosions:
            explosion.update(dt_ms)
            explosion.draw(internal_surface, camera)

        # Optionally, remove explosions that have no particles left:
        explosions = [e for e in explosions if e.particles]

        # Draw decorative showers (gold rain / confetti)
        for shower in decorative_showers:
            shower.update(dt_ms)
            shower.draw(internal_surface, camera)
        decorative_showers = [s for s in decorative_showers if not s.finished]

        # 다음 자동 랜덤 이벤트까지 남은 시간 (단일 타이머)
        hud.set_event_timers([("NEXT EVENT", (random_event_interval - (current_time - last_random_event)) / 1000)])

        # Draw HUD
        hud.draw(internal_surface)

        # Scale internal surface to fit the resized window (smoothscale to avoid pixelation when scaling)
        if (window_width, window_height) == (INTERNAL_WIDTH, INTERNAL_HEIGHT):
            scaled_surface.blit(internal_surface, (0, 0))
        else:
            pygame.transform.smoothscale(internal_surface, (window_width, window_height), scaled_surface)
        screen.blit(scaled_surface, (0, 0))

        # Save progress
        if current_time - last_save_progress >= save_progress_interval:
            # Save the game state or progress here
            print("Saving progress...")
            last_save_progress = current_time
            # Save progress to logs folder
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "progress.txt", "a+") as f:
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | ")
                f.write(f"Y: {-int(pickaxe.body.position.y // BLOCK_SIZE)} ")
                f.write(f"coal: {hud.amounts['coal']} ")
                f.write(f"iron: {hud.amounts['iron_ingot']} ")
                f.write(f"gold: {hud.amounts['gold_ingot']} ")
                f.write(f"copper: {hud.amounts['copper_ingot']} ")
                f.write(f"redstone: {hud.amounts['redstone']} ")
                f.write(f"lapis: {hud.amounts['lapis_lazuli']} ")
                f.write(f"diamond: {hud.amounts['diamond']} ")
                f.write(f"emerald: {hud.amounts['emerald']} \n")

        # Update the display
        pygame.display.flip()
        clock.tick(FRAMERATE)  # Cap the frame rate

        if current_time - last_fps_print >= 2000:
            last_fps_print = current_time
            print(f"FPS: {clock.get_fps():.1f}")

        # Inside the main loop
        keys = pygame.key.get_pressed()

        # Handle TNT spawn (key T)
        if keys[pygame.K_t]:
            if not key_t_pressed:  # Only spawn if the key was not pressed in the previous frame
                new_tnt = Tnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                            texture_atlas, atlas_items, sound_manager)
                tnt_list.append(new_tnt)
            key_t_pressed = True
        else:
            key_t_pressed = False  # Reset the flag when the key is released

        # Handle MegaTNT spawn (key M)
        if keys[pygame.K_m]:
            if not key_m_pressed:  # Only spawn if the key was not pressed in the previous frame
                new_megatnt = MegaTnt(space, pickaxe.body.position.x, pickaxe.body.position.y - 100,
                                    texture_atlas, atlas_items, sound_manager)
                tnt_list.append(new_megatnt)
            key_m_pressed = True
        else:
            key_m_pressed = False  # Reset the flag when the key is released

    # Quit pygame properly
    pygame.quit()

    # Return exit code: 0 for user quit (close window), 1 for crash/error
    if user_quit:
        import sys
        sys.exit(0)  # Normal exit - user closed window
    else:
        import sys
        sys.exit(1)  # Abnormal exit - game crashed or error

game()
