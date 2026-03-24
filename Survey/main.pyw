import sys
import asyncio
from PyQt5.QtWidgets import QApplication
import qasync
from server import SurveyServer


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PG Survey Helper")
    app.setStyle("Fusion")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    server = SurveyServer()

    with loop:
        loop.run_until_complete(server.run())


if __name__ == "__main__":
    main()
