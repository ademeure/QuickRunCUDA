COMMIT_CHANGES = False

import subprocess
import sys
import os
import threading
import time
import logging
import git
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from threading import Timer
import openai

logging.basicConfig(filename='directory_watcher.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

class DirectoryChangeHandler(FileSystemEventHandler):
    def __init__(self, command, output_file, repo_path, openai_key=None, max_summary_length=200):
        self.command = command
        self.output_file = os.path.abspath(output_file)
        self.repo_path = os.path.abspath(repo_path)
        self.openai_key = openai_key
        self.max_summary_length = max_summary_length
        self.max_input_characters = None
        self.is_handling_event = False
        self.lock = threading.Lock()
        self.debounce_timer = None
        self.last_modified_path = None
        self.last_modified_time = None
        self.has_command_error = False

    def on_modified(self, event):
        if not isinstance(event, FileModifiedEvent):
            return

        event_path = os.path.abspath(event.src_path)
        git_path = os.path.join(self.repo_path, '.git')
        if event_path == self.output_file or git_path in event_path or event_path.endswith('directory_watcher.log'):
            return

        # Get the actual file modification time
        try:
            file_mod_time = os.path.getmtime(event_path)
            output_mod_time = os.path.getmtime(self.output_file)
        except OSError:
            return

        # Skip if this modification happened during command execution
        if file_mod_time <= output_mod_time:
            return

        self.last_modified_path = event.src_path
        self.last_modified_time = time.time()

        if self.debounce_timer:
            self.debounce_timer.cancel()

        self.debounce_timer = Timer(0.05, self.trigger_event_handler)
        self.debounce_timer.start()

    def trigger_event_handler(self):
        with self.lock:
            if not self.is_handling_event:
                self.is_handling_event = True
                threading.Thread(target=self.handle_event).start()

    def handle_event(self):
        # Run the given command and stream the output in real-time
        with open(self.output_file, 'w') as f:
            self.has_command_error = False
            with self.lock:
                self.is_handling_event = True
            process = subprocess.Popen(f"bash -c \"{self.command}\"", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, shell=True)

            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

            # Check if there are errors and handle them
            error_lines = [line for line in iter(process.stderr.readline, '')]
            if error_lines:
                print("\n-----\nERROR\n-----\n", flush=True)
                f.write("\n-----\nERROR\n-----\n\n")
                f.flush()
                for line in error_lines:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()
                self.has_command_error = True

            process.stdout.close()
            process.stderr.close()
            process.wait()

        self.commit_changes()
        # Allow time for filesystem events to settle before re-enabling event handling
        time.sleep(0.05)

        # Re-enable event handling
        with self.lock:
            self.is_handling_event = False

    def commit_changes(self):
        if not COMMIT_CHANGES:
            return

        # Print git diff after running the command
        repo = git.Repo(self.repo_path)
        try:
            diff = repo.git.diff(':!directory_watcher.log')
        except git.exc.GitCommandError:
            diff = repo.git.diff()
        commit_message = f"Auto-commit: Change in {self.last_modified_path}"
        # Optionally create a summary of the diff using OpenAI API
        if self.openai_key and diff:
            client = openai.OpenAI(api_key=self.openai_key)

            # Truncate diff to max_input_characters
            if len(diff) > self.max_input_characters:
                diff = diff[:self.max_input_characters] + "\n[TRUNCATED DIFF]"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"Summarize the following git diff using bullet points (max {self.max_summary_length} characters, much less if trivial change). The 'output' file contains the command line output of running [{self.command}]. Highlight significant differences in output if any. If the output ends with a performance number WITH a unit (e.g. 300 ms or 3200 cycles or 50 TFLOPS - it MUST include the unit, if there is no unit it is NOT performance and should NOT be at the start of the first line) then *ALWAYS* include it at the START of the commit summary. The first line should NEVER be a bullet point, it should ALWAYS be be a minimal summary of the whole thing which should start with performance ONLY IF it is present at the end of the output WITH a unit like ms/fps/cycles (show it exactly as printed in the output file AND include a summary of the commit right after, e.g. '500.37 ms: doubled number of operations').\n===\n{diff}"
                }],
                max_tokens=self.max_summary_length // 4
            )
            summary = response.choices[0].message.content
            print("------------------------------------------------------------")
            print("DIFF SUMMARY:")
            print(summary)
            print("------------------------------------------------------------")
            commit_message = f"[AUTO {int(len(diff) / 1000)}K] {summary}"
        try:
            # Keep directory_watcher.log locally but ignored by git
            try:
                # Remove from git index but keep locally
                repo.git.rm('--cached', 'directory_watcher.log')
                # Add to .gitignore if not already there
                gitignore_path = os.path.join(self.repo_path, '.gitignore')
                if os.path.exists(gitignore_path):
                    with open(gitignore_path, 'r') as f:
                        if 'directory_watcher.log' not in f.read():
                            with open(gitignore_path, 'a') as f:
                                f.write('\ndirectory_watcher.log\n')
                else:
                    with open(gitignore_path, 'w') as f:
                        f.write('directory_watcher.log\n')
            except git.exc.GitCommandError:
                pass  # File isn't tracked by git

            repo.git.add(A=True)
            if self.has_command_error:
                commit_message = "[ERROR] " + commit_message
            repo.index.commit(commit_message)
            repo.remote(name='origin').push()
        except git.exc.GitCommandError as e:
            print(f"Error during Git operations: {e}")
            logging.error(f"Error during Git operations: {e}")


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: ./directory_watcher.py <directory_to_watch> <command_to_run> <output_file> [max_summary_length] [max_input_characters]")
        print("Note: Set OPENAI_API_KEY environment variable to enable OpenAI-powered commit messages")
        sys.exit(1)

    directory_to_watch = sys.argv[1]
    command_to_run = sys.argv[2]
    output_file = sys.argv[3]
    max_summary_length = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    max_input_characters = int(sys.argv[5]) if len(sys.argv) > 5 else 10000

    # Create empty output file if it does not exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write("")

    repo_path = os.path.abspath(directory_to_watch)
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    event_handler = DirectoryChangeHandler(command_to_run, output_file, repo_path, openai_key=openai_api_key, max_summary_length=max_summary_length)
    event_handler.max_input_characters = max_input_characters
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=True)

    observer.start()
    print(f"Watching directory: {directory_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        observer.stop()
        observer.join()
        if event_handler.is_handling_event:
            print("Waiting for event handling to complete...")
            time.sleep(1.5)


if __name__ == "__main__":
    main()
