import importlib.machinery
import importlib.util
import random
import sys
import types
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Digits,
    Footer,
    Header,
    Label,
    Log,
    RichLog,
    Static,
)

from glados_ui.text_resources import aperture, help_text, login_text, recipe

# This ugly stuff is necessary because there is a `glados` module as well as a `glados`
# package, so a normal `import glados` does the wrong thing.  If/when this is fixed
# in the `glados` module this can be simplieifed
loader = importlib.machinery.SourceFileLoader("glados", "./glados.py")
glados = types.ModuleType(loader.name)
loader.exec_module(glados)


# Custom Widgets


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    def on_mount(self) -> None:
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        if (text := event.text) != "\n":
            self.write(text.rstrip().replace("DEBUG", "[red]DEBUG[/]"))


class ScrollingBlocks(Log):
    """A widget for displaying random scrolling blocks."""

    BLOCKS = "⚊⚌☰𝌆䷀"
    DEFAULT_CSS = """
    ScrollingBlocks {
        scrollbar_size: 0 0;
        overflow-x: hidden;
    }"""

    def _animate_blocks(self) -> None:
        # Create a string of blocks of the right length, allowing
        # for border and padding
        random_blocks = " ".join(
            random.choice(self.BLOCKS) for _ in range(self.size.width - 8)
        )
        self.write_line(f"{random_blocks}")

    def on_show(self) -> None:
        self.set_interval(0.18, self._animate_blocks)


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = "",
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        *args: str,
        **kwargs: str,
    ) -> None:
        super().__init__(*args, *kwargs)
        self._text = text
        self.__id = id
        self._speed = speed
        self._repeat = repeat

    def compose(self) -> ComposeResult:
        self._static = Static()
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        return (self._text[:i] + "[blink]▃[/]" for i in range(len(self._text) + 1))

    def on_mount(self) -> None:
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Get and display the next character."""
        try:
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            self._static.update(next(self._iter_text))
        except StopIteration:
            if self._repeat:
                self._iter_text = self._get_iterator()


# Screens


class SplashScreen(Screen):
    """Splash screen shown on startup."""

    with open(Path("./glados_ui/images/splash.ansi")) as f:
        SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

    def compose(self) -> ComposeResult:
        with Container( id="splash_logo_container"):
            yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self):
        """Keep the screen scrolled to the bottom"""
        self.set_interval(0.5, self.scroll_end)

    def on_key(self, event: events.Key) -> None:
        """An  key is pressed."""
        # fire her up.....
        if event.key == 'q':
            app.action_quit()
        self.dismiss()
        app.start_glados()


class HelpScreen(ModalScreen):
    """The help screen. Possibly not that helpful."""

    BINDINGS = [("escape", "app.pop_screen", "Close screen")]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        yield Container(
            Typewriter(help_text, id="help_text"),
            id="help_dialog"
            )

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        dialog.border_subtitle = "[blink]Press Esc key to continue[/]"



# The App

class GladosUI(App):
    """The main app class for the GlaDOS ui."""

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(
            key="question_mark",
            action="help",
            description="Help",
            key_display="?",
        ),
    ]
    CSS_PATH = "glados_ui/glados.tcss"

    ENABLE_COMMAND_PALETTE = False

    TITLE = "GlaDOS v 1.09"

    SUB_TITLE = "(c) 1982 Aperture Science, Inc."

    with open(Path("./glados_ui/images/logo.ansi")) as f:
        LOGO_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")

    def compose(self) -> ComposeResult:
        """Generate the basic building blocks of the ui."""
        # It would be nice to have the date in the header, but see:
        # https://github.com/Textualize/textual/issues/4666
        yield Header(show_clock=True)

        with Container(id="body"):  # noqa: SIM117
            with Horizontal():
                yield (Printer(id="log_area"))
                with Container(id="utility_area"):
                    typewriter = Typewriter(recipe, id="recipe", speed=0.01, repeat=True)
                    yield typewriter

        yield Footer()

        # Blocks are displayed in a different layer, and out of the normal flow
        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="text_block", classes="block"):
                yield Digits("2.67")
                yield Digits("1002")
                yield Digits("45.6")
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

    def on_load(self) -> None:
        """Called when the app has been started but before the terminal is app mode."""
        # Cause logger to print all log text. Printed text can then be  captured
        # by the main_log widget
        logger.remove()
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>"
            "{level: <8}</level> | {name}:{function}:{line} - {message}"
        )
        logger.add(print, format=fmt)

    def on_mount(self) -> None:
        """Main screen is about to be shown."""
        # Display the splash screen for a few moments
        self.push_screen(SplashScreen())

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    def on_key(self, event: events.Key) -> None:
        """ "A key is pressed."""
        logger.debug(f"Pressed {event.character}")
        logger.info("some warning")

    def action_quit(self) -> None: # type: ignore
        """Bye bye."""
        # self.glados.cancel()
        self.exit(0)

    def start_glados(self):
        self.glados = self.run_worker(glados.start, exclusive=True, thread=True)
        pass


if __name__ == "__main__":
    try:
        app = GladosUI()
        app.run()
    except KeyboardInterrupt:
        sys.exit()
