#!/usr/bin/env python
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
import argparse

import os
import time
import subprocess
from pathlib import Path

class SubprocessController:
    def __init__(self, subprocess_popen):
        self.cmd_pipe_path = "/tmp/gitandrun_cmd"
        self.resp_pipe_path = "/tmp/gitandrun_resp"
        self.process = subprocess.Popen(subprocess_popen.split(' '))
        while not (os.path.exists(self.cmd_pipe_path) and os.path.exists(self.resp_pipe_path)):
            time.sleep(0.1)

    def send_command(self, args, read_response=True):
        cmd = " ".join(str(arg) for arg in args)
        with open(self.cmd_pipe_path, "w") as f:
            f.write(cmd)
        if read_response:
            with open(self.resp_pipe_path, "r") as f:
                return f.read()
        return None

    def __del__(self):
        try: self.send_command(["exit"], False)
        except: pass
        self.process.terminate()
        self.process.wait()

logging.basicConfig(filename='directory_watcher.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

class DirectoryChangeHandler(FileSystemEventHandler):
    def __init__(self, command, output_file, repo_path, openai_key=None, max_summary_length=200,
                 max_input_characters=10000, auto_commit=False, auto_push=False,
                 use_ai_summary=False, dry_run=False, subprocess_args_file=None):
        self.command = command
        self.output_file = os.path.abspath(output_file)
        self.repo_path = os.path.abspath(repo_path)
        self.openai_key = openai_key
        self.max_summary_length = max_summary_length
        self.max_input_characters = max_input_characters
        self.is_handling_event = False
        self.lock = threading.Lock()
        self.debounce_timer = None
        self.last_modified_path = None
        self.last_modified_time = None
        self.has_command_error = False
        self.auto_push = auto_push
        self.auto_commit = auto_commit
        self.dry_run = dry_run
        self.use_ai_summary = use_ai_summary
        self.subprocess_args_file = subprocess_args_file
        self.subprocess_controller = SubprocessController(command) if subprocess_args_file else None

    def __del__(self):
        if self.subprocess_controller:
            del self.subprocess_controller

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
        with open(self.output_file, 'w') as f:
            self.has_command_error = False
            with self.lock:
                self.is_handling_event = True

            if self.subprocess_controller:
                try:
                    with open(self.subprocess_args_file, 'r') as args_file:
                        args = args_file.read().strip().split()
                    response = self.subprocess_controller.send_command(args)
                    print(response, end='', flush=True)
                    f.write(response)
                    f.flush()
                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    f.write(f"Error: {e}\n")
                    self.has_command_error = True
            else:
                # Original subprocess behavior
                process = subprocess.Popen(f"bash -c \"{self.command}\"", stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1, shell=True)

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
        if not self.auto_commit:
            return

        # Print git diff after running the command
        repo = git.Repo(self.repo_path)
        try:
            diff = repo.git.diff(':!directory_watcher.log')
            if not diff:
                print("No changes to commit.")
                return
        except git.exc.GitCommandError:
            diff = repo.git.diff()
        commit_message = f"Auto-commit: Change in {self.last_modified_path}"

        # Optionally create a summary of the diff using OpenAI API
        if self.use_ai_summary and self.openai_key and diff:
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

        if self.has_command_error:
            commit_message = "[ERROR] " + commit_message
        if self.dry_run:
            print("\n--- DRY RUN - Would commit the following changes ---")
            print(f"Commit message: {commit_message}")
            print(f"Changes:\n{diff[:500]}{'...' if len(diff) > 500 else ''}")
            print("--- End of dry run ---\n")
            return

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
            repo.index.commit(commit_message)

            # Only push if auto_push is enabled
            if self.auto_push:
                repo.remote(name='origin').push()
            else:
                print("Auto-push disabled. Changes committed but not pushed.")

        except git.exc.GitCommandError as e:
            print(f"Error during Git operations: {e}")
            logging.error(f"Error during Git operations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Watch a directory and run a command when files change.')
    parser.add_argument('auto_commit', choices=['0', '1', 'no','commit'],
                        help='Whether to automatically commit changes')
    parser.add_argument('auto_push', choices=['0', '1', 'no','push'],
                        help='Whether to automatically push commits to remote')
    parser.add_argument('use_ai_summary', choices=['0', '1', 'no', 'summary'],
                        help='Whether to use OpenAI to generate commit summaries')
    parser.add_argument('directory_to_watch', help='Directory to watch for changes')
    parser.add_argument('command_to_run', help='Command to run when changes are detected (or at init for subprocess mode)')
    parser.add_argument('output_file', help='File to write command output to')
    parser.add_argument('--max-summary-length', type=int, default=400,
                        help='Maximum length of commit summary')
    parser.add_argument('--max-input-characters', type=int, default=10000,
                        help='Maximum number of characters to send to OpenAI API')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be committed without making changes')
    parser.add_argument('--subprocess-args-file',
                        help='Use subprocess server instead and read arguments from given file')

    # For backward compatibility with positional args
    if len(sys.argv) >= 7 and not any(arg.startswith('--') for arg in sys.argv[1:7]):
        args = parser.parse_args()
        auto_commit = args.auto_commit.lower() in ('1', 'commit')
        auto_push = args.auto_push.lower() in ('1', 'push')
        use_ai_summary = args.use_ai_summary.lower() in ('1', 'summary')
        directory_to_watch = args.directory_to_watch
        command_to_run = args.command_to_run
        output_file = args.output_file
        max_summary_length = args.max_summary_length
        max_input_characters = args.max_input_characters
        dry_run = args.dry_run
        subprocess_args_file = args.subprocess_args_file
    else:
        print("Error: Invalid arguments")
        parser.print_help()
        sys.exit(1)

    # Check if OpenAI API key is set when AI summary is enabled
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if use_ai_summary and not openai_api_key:
        print("Error: OpenAI API key (OPENAI_API_KEY) must be set when AI summary is enabled.")
        print("Please set the environment variable and try again.")
        sys.exit(1)

    # Create empty output file if it does not exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write("")

    repo_path = os.path.abspath(directory_to_watch)

    # Create event handler with all parameters
    event_handler = DirectoryChangeHandler(
        command=command_to_run,
        output_file=output_file,
        repo_path=repo_path,
        openai_key=openai_api_key if use_ai_summary else None,
        max_summary_length=max_summary_length,
        max_input_characters=max_input_characters,
        auto_commit=auto_commit,
        auto_push=auto_push,
        use_ai_summary=use_ai_summary,
        dry_run=dry_run,
        subprocess_args_file=subprocess_args_file
    )

    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=True)

    # Print status message about git operations
    print(f"Git operations: {'Commit ' if auto_commit else 'No commit '}{'+ Push' if auto_push else '(no push)'}")
    print(f"AI commit summaries: {'Enabled' if use_ai_summary else 'Disabled'}")
    print(f"Watching directory: {directory_to_watch}")

    observer.start()
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
