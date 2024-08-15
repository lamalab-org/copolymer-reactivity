from pygetpapers import Pygetpapers


pygetpapers_call=Pygetpapers()
pygetpapers_call.run_command(query='"Copolymerization" AND "reactivity"', limit=2500, output='files_pygetpaper', xml=True, update=True)