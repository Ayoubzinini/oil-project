$act=0
while ($act -eq 0) {
	cd E:\scripts\oil-project
	"0 : Modify"
	"1 : Run"
	$m_or_r= Read-Host -Prompt "Modify or Run?"
	if ($m_or_r -eq 0){
		code .
	}
	elseif ($m_or_r -eq 1){
		$dir = get-childitem
		$pyfiles = @()
		foreach ($file in $dir){
			if ($file.fullname.ToString() -like "*.py") {
				$pyfiles += $file.fullName.ToString()
			}
		}
		"Your python files : "
		$i = 0
		foreach ($file in $pyfiles){
			$i.ToString() + " : " + $file.ToString().substring(23)
			$i += 1
		}
		$choosed = Read-Host -Prompt "Choose the index of the target file to run it"
		python $pyfiles[$choosed].substring(23)
	}
	$act = Read-Host -Prompt "Input 0 to continue or another value to go out"
}
