<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$db = db_connect();

if (isset($_GET['profile'])){
	$doctorid = $_GET['profile'];
	$doctor_profile = new doctors_profile($doctorid);
	$_SESSION['doctor_profile'] = $doctor_profile;
	$doctor_profile->display_profile();
	
} else if (isset($_GET['delete'])){
	$doctorid = $_GET['delete'];
	doctor::delete_doctor($db, $doctorid);
	echo "<meta http-equiv=refresh content=0;URL='/db_entry/doctors_profile.php'";
} else {
	$doctor_profile = $_SESSION['doctor_profile'];
	$db = db_connect();
	$doctor_profile->db = $db;
	if (isset($_GET['schedule'])){
		$doctorid = $doctor_profile->doctorid;
		display_doctors_schedule_form($db, $doctorid);
	} else {
		$doctor_profile->display_profile_edit_form();
	}
}

$db->close();
do_footer();
?>