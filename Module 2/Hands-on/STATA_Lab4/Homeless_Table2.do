
cd "E:\Desktop\STATA Lab\STATA_Lab4"


* Global macros:
global zs i.dow i.month i.timeofmonth i.year*finaldebtneed2 i.year*finaldebtneed3 i.year*finaldebtneed4 i.year*finaldebtneed5 i.yr_quarter*finaldebtneed2 i.yr_quarter*finaldebtneed3 i.yr_quarter*finaldebtneed4 i.yr_quarter*finaldebtneed5 veteran_zeros i.year*missing_callercontribution i.year*missing_totaldebtneed housingsubsidy_zeros housingsubsidy_m need_more_than_1mo_rent_zeros need_more_than_1mo_rent_zeros_m inc2Xpov_zeros inc2Xpov_m has_ssn_zeros has_ssn_m livingsit_own 
global zs_rent i.dow i.month i.timeofmonth i.year*finaldebtneedrent2 i.year*finaldebtneedrent3 i.year*finaldebtneedrent4 i.year*finaldebtneedrent5 i.yr_quarter*finaldebtneedrent2 i.yr_quarter*finaldebtneedrent3 i.yr_quarter*finaldebtneedrent4 i.yr_quarter*finaldebtneedrent5 veteran_zeros i.year*missing_callercontribution i.year*missing_totaldebtneed housingsubsidy_zeros need_more_than_1mo_rent rank inc2Xpov_zeros inc2Xpov_m has_ssn_zeros has_ssn_m livingsit_own 
global zs_sd i.dow i.month i.timeofmonth i.year*finaldebtneed2 i.year*finaldebtneed3 i.year*finaldebtneed4 i.year*finaldebtneed5 i.yr_quarter*finaldebtneed2 i.yr_quarter*finaldebtneed3 i.yr_quarter*finaldebtneed4 i.yr_quarter*finaldebtneed5 veteran_zeros i.year*missing_callercontribution i.year*missing_totaldebtneed housingsubsidy_zeros housingsubsidy_m inc2Xpov_zeros inc2Xpov_m has_ssn_zeros has_ssn_m livingsit_own 
global xs gender_zeros race1_zeros o.race2_zeros race3_zeros hispanic_zeros age_zeros adultsinhousehold_zeros minorsinhousehold_zeros hs_degree_zeros part_rate_zeros unemployment_rate_zeros median_age_zeros mnthly_housing_cost_thous_zeros med_hshld_income_thous_zeros prcnt_black_zeros o.prcnt_white_zeros  prcnt_otherrace_zeros  benefitloss_zeros cantaffordbills_zeros exitingsharedhousing_zeros fleeingabuse_zeros jobloss_zeros monthlyincomeinthousands_zeros exiting_shelter_zeros receivingincomesource1_zeros receivingincomesource2_zeros receivingincomesource3_zeros receivingincomesource4_zeros receivingincomesource5_zeros receivingincomesource6_zeros receivingincomesource7_zeros receivingbenefits1_zeros gender_m race1_m o.race2_m race3_m hispanic_m age_m adultsinhousehold_m minorsinhousehold_m hs_degree_m part_rate_m unemployment_rate_m median_age_m mnthly_housing_cost_thous_m med_hshld_income_in_thousands_m prcnt_black_m o.prcnt_white_m  prcnt_otherrace_m  benefitloss_m cantaffordbills_m exitingsharedhousing_m fleeingabuse_m jobloss_m monthlyincomeinthousands_m exiting_shelter_m receivingincomesource1_m receivingincomesource2_m receivingincomesource3_m receivingincomesource4_m receivingincomesource5_m receivingincomesource6_m receivingincomesource7_m receivingbenefits1_m in_shelter_in_window rank livingsit_rent livingsit_shared  


* Table 2
use "Homeless.dta", clear

// ssc install estout

local sub_list " "main" "homogeneous" "rent" "securitydeposit" "

foreach j in `sub_list' {

* Main sample
	if "`j'"=="main" {
		foreach i of numlist 3 6 {
		eststo clear
				eststo: xi: reg shelter_`i'_mos samedayfunds $zs $xs securitydeposit, cluster(zipcode)
			esttab using table2_mainreg.csv, plain star se ml("`i'") keep(samedayfunds) stats(N)
		}
		eststo clear
		eststo: xi: reg days_total_6_mos samedayfunds $zs $xs securitydeposit, cluster(zipcode)
		esttab using table2_maindays6mos_reg.csv, plain star se keep(samedayfunds) stats(N)
	}
	
* Subsample
	else {
		foreach i of numlist 3 6 {
			eststo clear
			eststo: xi: reg shelter_`i'_mos samedayfunds $zs $xs securitydeposit if `j'==1, cluster(zipcode)
			esttab using table2_`j'.csv, plain star se ml("`i'") keep(samedayfunds) stats(N)
		}
		eststo clear
		eststo: xi: reg days_total_6_mos samedayfunds $zs $xs securitydeposit if `j'==1, cluster(zipcode)
		esttab using table2_`j'_days.csv, plain star se keep(samedayfunds) stats(N)
	
		if "`j'"=="rent" {
	
		* Rent callers, below median income
		foreach i of numlist 3 6 {
			eststo clear
			eststo: xi: reg shelter_`i'_mos samedayfunds $zs $xs securitydeposit if `j'==1 & below_med_income_rent==1, cluster(zipcode)
			esttab using table2_`j'_belowmed.csv, plain star se ml("`i'") keep(samedayfunds) stats(N)
		}
		eststo clear
		eststo: xi: reg days_total_6_mos samedayfunds $zs $xs securitydeposit if `j'==1 & below_med_income_rent==1, cluster(zipcode)
		esttab using table2_`j'_belowmed_days.csv, plain star se keep(samedayfunds) stats(N)

		* Rent callers, above median income
		foreach i of numlist 3 6 {
			eststo clear
			eststo: xi: reg shelter_`i'_mos samedayfunds $zs $xs securitydeposit if `j'==1 & below_med_income_rent==0, cluster(zipcode)
			esttab using table2_`j'_abovemed.csv, plain star se ml("`i'") keep(samedayfunds) stats(N)
		}
		eststo clear
		eststo: xi: reg days_total_6_mos samedayfunds $zs $xs securitydeposit if `j'==1 & below_med_income_rent==0, cluster(zipcode)
		esttab using table2_`j'_abovemed_days.csv, plain star se keep(samedayfunds) stats(N)
		
		}
	
	}

}
