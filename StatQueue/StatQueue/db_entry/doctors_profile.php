<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$db = db_connect();

$user = check_valid_user();

if ($user == 'user' && isset($_GET['clinicid'])){
	#clinicid is sent from search_result; regular user is trying to 
	#see the list of doctors of this clinic.
	$clinicid = $_GET['clinicid'];
	$clinic = new clinic($db, $clinicid);
	$_SESSION['saved_clinic'] = $clinic;

} else {
	#admin is following up her saved clinic.(or regular user has refreshed current page.)
	$clinic = $_SESSION['saved_clinic'];
	$clinic->db = $db;
	if (isset($_GET['refresh'])){
		$clinic->reset_avail_doctors();
	}
}

$clinic->display_doctors($user);

$db->close();
do_footer();
?>