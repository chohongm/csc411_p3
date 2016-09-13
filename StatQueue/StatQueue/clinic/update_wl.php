<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");
session_start();

if (isset($_POST['del_me'])){
	$del_list = $_POST['del_me'];
	$clinicid = $_POST['clinicid'];
	$wl_length = $_POST['wl_length'];	
	$db = db_connect();
	
	foreach ($del_list as $wl_id){
		clinic::unlist_patient($db, $clinicid, $wl_id, $wl_length);
	}
	
	include("$DOCUMENT_ROOT/clinic/current_wl.php");
		
} else if(isset($_POST['order1']) && isset($_POST['order2'])){
	$order1 = $_POST['order1'];
	$order2 = $_POST['order2'];

	$db = db_connect();
	$myclinic = $_SESSION['myclinic'];
	$myclinic->db = $db;
	
	if (valid_order($myclinic->rows, $order1, $order2)){
		$myclinic->change_wl_order($myclinic->rows, $order1, $order2);
	}
	
	include("$DOCUMENT_ROOT/clinic/current_wl.php");
	
} else {
	throw new Excpetion("Please provide every necessary information
						 in order to update your clinic's waitlist.");
}

?>