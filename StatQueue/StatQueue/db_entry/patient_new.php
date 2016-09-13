<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();

  $user = check_valid_user();
  if ($user != 'user'){
  	
  	$info_list = array_slice($_POST, 0, 10);
  	if (!filled_out($info_list)){
  		throw new Exception('Please enter every required detail.');
  	}
  	
  	if (!valid_birthdate($_POST['birthdate'])){
  		throw new Exception("Invalid birthdate.");
  	}
  	
  	if (!valid_email($_POST['email'])){
  		throw new Exception("The email you have entered is not valid.");
  	}
  	
  	if (!valid_postalcode($_POST['postalcode'])){
  		throw new Exception("Postal code you have entered is not valid.");
  	}
	
  	$db = db_connect();
  	$myclinic = $_SESSION['myclinic'];
  	$myclinic->db = $db;
  	$fullname = $_POST['firstname']." ".$_POST['lastname'];
  	$info_list = array();
  	$info_list['fullname'] = $fullname;
  	  	
  	foreach ($_POST as $key => $value){
  		if ($key != 'lastname' && $key != 'firstname' && $key != 'treatment' && $key != 'doctor'){
  			$info_list[$key] = $value;
  		}
  	}
  	 
	echo "<h2>Patient Entry Result</h2>";
	
	if (!get_magic_quotes_gpc()){
		foreach ($info_list as $value){
			$value = addslashes($value);
		}
		addslashes($_POST['treatment']);
		addslashes($_POST['doctor']);
	}
	
	@ $db = db_connect();
	$patient = new patient($info_list);
	
	if ($patient_info = patient::lookup_patient($db, $info_list['healthcardnumber'])){
		$patientid = $patient_info['patientid'];
		echo '<br />Patient is already registered. Patient ID is '.$patientid.".<br/>";
	} else {
		$patient->register_patient($db);
		$patientid = $patient->patientid;
	}
	
	if ($myclinic->lookup_patient($patientid)){
		throw new Exception ("patient is already added to your clinic's waitlist.");
	}
	
	foreach ($info_list as $key => $value){
		if ($key == 'fullname' || $key == 'patientname' || $key == 'birthdate' ||
				$key == 'gender' || $key == 'medicalhistory'){
			$wl_list[$key] = $value;
		}
	}
	$wl_list['treatment'] = $_POST['treatment'];
	$wl_list['doctor'] = find_term_like($db, 'doctorname', 'doctors', $_POST['doctor']);
	if (isset($patientid)){
		$wl_list['patientid'] = $patientid;
	}
	
	$myclinic->list_patient($wl_list);
	
	$query = "select * from waitlist where patientid = ".$patientid;
	$result = mysqli_query($db, $query);
	$row = mysqli_fetch_row($result);
	array_push($myclinic->rows, $row);
	$_SESSION['saved_clinic'] = $myclinic;
	
	echo "<br/>Patient is added to your waitlist.<br/><br/>
			<a href='/clinic/current_wl.php'>Return to waitlist page</a>";
	
	$db->close();
  }
  do_footer();
?>