"""Voicemail compliance detection runner script.

Scans a demo directory of WAV files and attempts to drop a provided
voice_mail.wav into detected voicemail drop points using VoicemailDropper.
Saves a results file under output/results.txt and prints a summary to stdout.
"""

import os
from voicemail_dropper import VoicemailDropper


def main():
    """Entry point for the voicemail compliance detection runner.

    - Verifies demo directory and optional voice_mail.wav presence.
    - Instantiates VoicemailDropper with optional GITHUB_TOKEN.
    - Processes files in demo_files producing dropped outputs in 'output'.
    - Prints a final summary table and writes results to output/results.txt.
    """
    print('='*50)
    print('VOICEMAIL COMPLIANCE DETECTION - MODULAR')
    print('='*50)
    demo_dir = 'demo_files'
    if not os.path.exists(demo_dir):
        print(f"Error: '{demo_dir}' not found. Create the folder and add .wav files.")
        return
    voice_mail = 'voice_mail.wav'
    if not os.path.exists(voice_mail):
        print(f"Warning: '{voice_mail}' not found. Please add your voice_mail.wav in project root.\nThe system will still run but cannot create dropped files without it.")
    dropper = VoicemailDropper(github_token=os.getenv('GITHUB_TOKEN'))
    results = dropper.process_directory(demo_dir, voice_mail_path=voice_mail, output_dir='output')
    print('\n' + '='*50)
    print('FINAL RESULTS')
    print('='*50)
    print(f"{'File':<25} {'Drop Time':<12} {'Trigger':<30} {'Status':<10}")
    print('-'*80)
    for fn, r in results.items():
        ts = f"{r['timestamp']:.2f}s" if r['timestamp'] else 'N/A'
        print(f"{fn:<25} {ts:<12} {r['reason']:<30} {r['status']:<10}")
    out_results = 'output/results.txt'
    with open(out_results, 'w') as f:
        f.write('Voicemail Drop Timestamps\n')
        f.write('='*40 + '\n')
        for fn, r in results.items():
            ts = f"{r['timestamp']:.2f}" if r['timestamp'] else 'N/A'
            f.write(f"{fn}: {ts} seconds ({r['reason']}) -> {r['status']}\n")
    print(f"\nResults saved to {out_results}")


if __name__ == '__main__':
    main()
