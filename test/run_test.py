#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import torch

TESTS = [
    'autograd',
    'cpp_extensions',
    'cuda',
    'dataloader',
    'distributed',
    'distributions',
    'indexing',
    'jit',
    'legacy_nn',
    'multiprocessing',
    'nccl',
    'nn',
    'optim',
    'sparse',
    'torch',
    'utils',
]

WINDOWS_BLACKLIST = [
    'distributed',
]

DISTRIBUTED_TESTS_CONFIG = {
    'tcp': {
        'WORLD_SIZE': '3'
    },
    'gloo': {
        'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3'
    },
    'nccl': {
        'WORLD_SIZE': '2'
    },
    'mpi': {},
}


def print_to_stderr(message):
    print(message, file=sys.stderr)


def shell(command, cwd):
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.call(
        shlex.split(command), universal_newlines=True, cwd=cwd) == 0


def get_shell_output(command):
    return subprocess.check_output(shlex.split(command)).decode().strip()


def run_test(python, test_module, test_directory, options):
    verbose = '--verbose' if options.verbose else ''
    return shell('{} -m unittest {} {}'.format(python, verbose, test_module),
                 test_directory)


def test_cpp_extensions(python, test_module, test_directory, options):
    if not shell('{} setup.py install --root ./install'.format(python),
                 os.path.join(test_directory, 'cpp_extensions')):
        return False

    python_path = os.environ.get('PYTHONPATH', '')
    try:
        cpp_extensions = os.path.join(test_directory, 'cpp_extensions')
        if sys.platform == 'win32':
            install_directory = os.path.join(cpp_extensions, 'install')
            install_directories = get_shell_output(
                "where -r \"{}\" *.pyd".format(install_directory)).split('\r\n')

            assert install_directories, 'install_directory must not be empty'

            if len(install_directories) >= 1:
                install_directory = install_directories[0]

            install_directory = os.path.dirname(install_directory)
            split_char = ';'
        else:
            install_directory = get_shell_output(
                "find {}/install -name *-packages".format(cpp_extensions))
            split_char = ':'

        assert install_directory, 'install_directory must not be empty'
        install_directory = os.path.join(test_directory, install_directory)
        os.environ['PYTHONPATH'] = '{}{}{}'.format(install_directory,
                                                   split_char,
                                                   python_path)
        return run_test(python, test_module, test_directory, options)
    finally:
        os.environ['PYTHONPATH'] = python_path


def test_distributed(python, test_module, test_directory, options):
    mpi_available = subprocess.call('command -v mpiexec', shell=True) == 0
    if options.verbose and not mpi_available:
        print_to_stderr(
            'MPI not available -- MPI backend tests will be skipped')
    for backend, env_vars in DISTRIBUTED_TESTS_CONFIG.items():
        if backend == 'mpi' and not mpi_available:
            continue
        for with_init_file in {True, False}:
            tmp_dir = tempfile.mkdtemp()
            if options.verbose:
                with_init = ' with file init_method' if with_init_file else ''
                print_to_stderr(
                    'Running distributed tests for the {} backend{}'.format(
                        backend, with_init))
            os.environ['TEMP_DIR'] = tmp_dir
            os.environ['BACKEND'] = backend
            os.environ['INIT_METHOD'] = 'env://'
            os.environ.update(env_vars)
            if with_init_file:
                init_method = 'file://{}/shared_init_file'.format(tmp_dir)
                os.environ['INIT_METHOD'] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, 'barrier'))
                os.mkdir(os.path.join(tmp_dir, 'test_dir'))
                if backend == 'mpi':
                    mpiexec = 'mpiexec -n 3 --noprefix {}'.format(python)
                    if not run_test(mpiexec, test_module, test_directory,
                                    options):
                        return False
                elif not run_test(python, test_module, test_directory,
                                  options):
                    return False
            finally:
                shutil.rmtree(tmp_dir)
    return True


CUSTOM_HANDLERS = {
    'cpp_extensions': test_cpp_extensions,
    'distributed': test_distributed,
}


def parse_test_module(test):
    idx = test.find('.')
    return test[:idx if idx > -1 else None]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the PyTorch unit test suite',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information and test-by-test results')
    parser.add_argument(
        '-p', '--python', help='the python interpreter to execute tests with')
    parser.add_argument(
        '-c', '--coverage', action='store_true', help='enable coverage')
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests can be specified with module name, module.TestClass'
             ' or module.TestClass.test_method')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '-f',
        '--first',
        choices=TESTS,
        metavar='TESTS',
        help='select the test to start from (excludes previous tests)')
    parser.add_argument(
        '-l',
        '--last',
        choices=TESTS,
        metavar='TESTS',
        help='select the last test to run (excludes following tests)')
    parser.add_argument(
        '--ignore-win-blacklist',
        action='store_true',
        help='always run blacklisted windows tests')
    return parser.parse_args()


def get_python_command(options):
    if options.coverage:
        return 'coverage run --parallel-mode --source torch'
    elif options.python:
        return options.python
    else:
        return os.environ.get('PYCMD', 'python')


def find_test_index(test, selected_tests, find_last_index=False):
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            if not find_last_index:
                return idx
            else:
                found_idx = idx
        idx += 1
    return found_idx


def exclude_tests(exclude_list, tests, exclude_message=None):
    sel_tests = tests[:]
    for test in exclude_list:
        for t in sel_tests:
            if t.startswith(test):
                if exclude_message is not None:
                    print_to_stderr(('Excluding {} ' + exclude_message).format(t))
                tests.remove(t)
    return tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == 'win32' and not options.ignore_win_blacklist:
        selected_tests = exclude_tests(WINDOWS_BLACKLIST, selected_tests, 'on Windows')

    return selected_tests


def main():
    options = parse_args()
    python = get_python_command(options)
    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    if options.coverage:
        shell('coverage erase')

    for test in selected_tests:
        test_name = 'test_{}'.format(test)
        test_module = parse_test_module(test)

        print_to_stderr('Running {} ...'.format(test_name))
        handler = CUSTOM_HANDLERS.get(test_module, run_test)
        if not handler(python, test_name, test_directory, options):
            raise RuntimeError('{} failed!'.format(test_name))

    if options.coverage:
        shell('coverage combine')
        shell('coverage html')


if __name__ == '__main__':
    main()
