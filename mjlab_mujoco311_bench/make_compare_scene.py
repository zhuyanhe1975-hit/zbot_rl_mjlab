from pathlib import Path
import re
src=Path('model.xml').read_text()
body=re.search(r'    (<body name="robot/foot_0">.*?    </body>\n  </worldbody>)',src,re.S).group(1)
body=body[:body.rfind('    </body>\n  </worldbody>')+len('    </body>')]
# Keep one terrain body, duplicate robot with a visible x offset supplied by qpos.
ref=body.replace('robot/','ref/').replace('mesh="ref/','mesh="robot/')
native=body.replace('robot/','native/').replace('mesh="native/','mesh="robot/')
world='<worldbody>\n    <body name="terrain"><geom name="terrain" size="0 0 0.01" type="plane" material="groundplane" group="0" /></body>\n'+ref+'\n'+native+'\n  </worldbody>'
act=re.search(r'  <actuator>.*?  </actuator>',src,re.S).group(0)
act=act.replace('robot/','ref/')+'\n'+act.replace('robot/','native/')
contact=re.search(r'  <contact>.*?  </contact>',src,re.S).group(0)
contact=contact.replace('robot/','ref/')+'\n'+contact.replace('robot/','native/')
asset=re.search(r'  <asset>.*?  </asset>',src,re.S).group(0)
# The original mesh assets are shared; names remain robot/.
xml=re.sub(r'  <worldbody>.*?</worldbody>',world,src,flags=re.S)
xml=re.sub(r'  <contact>.*?</contact>',contact,xml,flags=re.S)
xml=re.sub(r'  <actuator>.*?</actuator>',act,xml,flags=re.S)
xml=re.sub(r'  <sensor>.*?</sensor>', '  <sensor></sensor>', xml, flags=re.S)
xml=re.sub(r'  <keyframe>.*?</keyframe>', '', xml, flags=re.S)
Path('compare_scene.xml').write_text(xml)
print('wrote compare_scene.xml',len(xml))
