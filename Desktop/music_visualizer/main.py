import os
import pygame
import tomllib

def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)

def main() -> None:
    config = load_config()
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    pygame.init()
    width, height = config["export"]["resolution"]
    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Music Visualizer")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        clock.tick(config["renderer"]["realtime_fps"])

    pygame.quit()

if __name__ == "__main__":
    main()
