sysuse nlsw88, clear

// Own program
program define jypark_tabstat
   levelsof `1', local(level)
   foreach i in `level' {
      local value_label : label(`1') `i'
      sum wage if `1' ==`i'
display _newline "summary of wage, `1' = `value_label'"
   }
end


/* Build-in program */

jypark_tabstat occupation
tabstat wage, by(occupation)

jypark_tabstat industry
tabstat wage, by(industry)

