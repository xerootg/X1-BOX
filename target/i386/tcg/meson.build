x87_lib_proj = subproject('x87', required: false)
if x87_lib_proj.found()
  x87_lib_dep = x87_lib_proj.get_variable('x87_dep')
else
  x87_lib_dep = disabler()
endif

i386_ss.add(when: 'CONFIG_TCG', if_true: files(
  'access.c',
  'bpt_helper.c',
  'cc_helper.c',
  'excp_helper.c',
  'fpu_helper.c',
  'fpu_helper_hard.c',
  'int_helper.c',
  'mem_helper.c',
  'misc_helper.c',
  'mpx_helper.c',
  'seg_helper.c',
  'tcg-cpu.c',
  'translate.c',
  'x87_shadow.c',
  'x87_lib_shim.cpp'), if_false: files('tcg-stub.c'))

i386_ss.add(when: 'CONFIG_TCG', if_true: x87_lib_dep)

subdir('system')
subdir('user')
