import runpy

def main():
        runpy.run_module('supporting_functions', run_name='__main__')
        runpy.run_module('plots_functions', run_name='__main__')
        runpy.run_module('dash_frontend', run_name='__main__')

if __name__ == "__main__":
    main()

