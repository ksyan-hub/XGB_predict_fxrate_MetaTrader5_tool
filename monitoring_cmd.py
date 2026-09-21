from config import cmd_config

class cmd_monitor:
    def exit_run(self, event):
        config = cmd_config()

        while True:
            user_input = input("終了コマンド = exit_run: ")

            if user_input.strip().lower() == config.exit_run:
                print(f"{config.exit_run}コマンドを受け付けました。処理を終了します")
                event = event.set()
                break