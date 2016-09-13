<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
session_start();

$db = db_connect();
$myclinic = $_SESSION['saved_clinic'];
$myclinic->db = $db;

if ($_POST['patientid'] != '' && $_POST['treatment'] != ''){
	$query = "select * from waitlist where patientid = ".$_POST['patientid']."
			  and clinicid = ".$myclinic->clinicid;
	$result = $db->query($query);
	
	if (mysqli_num_rows($result) > 0){
		throw new Exception('The patient you entered is already waitlisted in your clinic.');
	}
	
	$query_p = "select patientname, birthdate, gender, medicalhistory from patients
				where patientid = ".$_POST['patientid'];
	$result_p = $db->query($query_p);
	if ($result_p->num_rows == 0){
		throw new Exception("The patient ID you entered is invalid");
	}
	$row = $result_p->fetch_assoc();
	$info_list = array();
	foreach ($row as $key => $value){
		$info_list[$key] = $value;
	}
	
	$info_list['treatment'] = $_POST['treatment'];
	$info_list['doctor'] = find_term_like($db, 'doctorname', 'doctors', $_POST['doctor']);
	$info_list['patientid'] = $_POST['patientid'];
	
} else if ($_POST['lastname'] != '' && $_POST['firstname'] != '' && $_POST['birthdate'] != '' &&
		   $_POST['gender'] != '' && $_POST['medicalhistory'] != '' && $_POST['treatment'] != ''){
	
	$fullname = $_POST['firstname']." ".$_POST['lastname'];
	$query = "select * from waitlist where patientname = '".$fullname."'
		  and birthdate = '".$_POST['birthdate']."'
		  and clinicid = ".$myclinic->clinicid;
	$result = mysqli_query($db, $query);
	
	if (mysqli_num_rows($result) > 0){
		throw new Exception('The patient you entered is already waitlisted in your clinic.');
	}
	
	$info_list[0] = $fullname;
	foreach ($_POST as $key => $value){
		if ($key != 'lastname' && $key != 'firstname' && $key != 'patientid'){
			$info_list[$key] = $value;
		}
	}
} else {
	throw new Exception('Please enter every required detail.');
}

$myclinic->list_patient($info_list);

include("$DOCUMENT_ROOT/clinic/current_wl.php");
?>
