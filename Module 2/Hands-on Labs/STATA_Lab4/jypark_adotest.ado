
program define jypark_tabstat
   levelsof `1', local(level)
   foreach i in `level' {
      local value_label : label(`1') `i'
      display _newline "ado test `1' = `value_label'"
   }
end

exit
