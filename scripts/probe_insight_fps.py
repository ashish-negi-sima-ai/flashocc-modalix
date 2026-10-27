"""Measure decoded/rendered browser frames over wall time in Insight's viewer."""
import argparse
import json
import ssl
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright
from runtime_config import defaults, require

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--api',default=defaults()['insight_api'])
    args=parser.parse_args()
    require(parser,args.api,'--api','FLASHOCC_INSIGHT_API')
    def api(path):
        with urllib.request.urlopen(args.api+path,context=ssl._create_unverified_context(),timeout=10) as response:
            return json.load(response)
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        try:
            page=browser.new_page(ignore_https_errors=True,viewport={'width':1600,'height':1050})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            viewer=api('/api/viewer-url?mode=light&src=0')['url'];page.goto(viewer,wait_until='domcontentloaded')
            page.wait_for_function('Array.from(document.querySelectorAll("video")).some(v=>v.videoWidth===1536 && v.currentTime>0)',timeout=90000)
            page.wait_for_timeout(7000)
            def snapshot():
                return page.evaluate('''() => {
                    const v=Array.from(document.querySelectorAll('video')).find(v=>v.videoWidth===1536 && v.currentTime>0);
                    const q=v.getVideoPlaybackQuality();return {time_ms:performance.now(),frames:q.totalVideoFrames,
                    dropped:q.droppedVideoFrames,width:v.videoWidth,height:v.videoHeight};}''')
            first=snapshot();page.wait_for_timeout(20000);last=snapshot()
            duration=(last['time_ms']-first['time_ms'])/1000
            result=dict(passed=last['frames']>first['frames'] and not errors,
                elapsed_seconds=duration,browser_fps=(last['frames']-first['frames'])/duration,
                dropped_frames=last['dropped']-first['dropped'],first=first,last=last,errors=errors,viewer=viewer)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            page.screenshot(path=str(args.output.with_suffix('.png')),full_page=True)
            args.output.with_name('ingest.json').write_text(json.dumps(api('/api/ingest/stats'),indent=2)+'\n')
            args.output.with_name('egress.json').write_text(json.dumps(api('/api/egress/stats'),indent=2)+'\n')
            if not result['passed']:raise RuntimeError(result)
            print(json.dumps(result))
        finally:browser.close()

if __name__=='__main__':main()
