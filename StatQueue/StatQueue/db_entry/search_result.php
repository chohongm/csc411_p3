<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

@ $db = db_connect();

if (isset($_GET['postalcode'])){
	$first_two = substr($_GET['postalcode'], 0, 2);
	$regexp = '^'.$first_two.'[A-Z0-9]+$';
	$searchtype = "c.postalcode";
	$operator = "rlike";
	$searchterm = $regexp;
	display_search_results($db, $searchtype, $searchterm, $operator);
	
} else {
	if (isset($_POST['searchterm']) && isset($_POST['searchtype'])){
		$searchtype = $_POST['searchtype'];
		$searchterm = '%'.trim($_POST['searchterm']).'%';
		$_SESSION['searchtype'] = $searchtype;
		$_SESSION['searchterm'] = $searchterm;
	} else if(isset($_SESSION['searchterm']) && isset($_SESSION['searchtype'])){
		$searchtype = $_SESSION['searchtype'];
		$searchterm = $_SESSION['searchterm'];
	}
	
	if (!isset($searchterm)) {
		echo 'You have not entered any search detail. Please try again.';
		exit;
	}
	
	if (!get_magic_quotes_gpc()){
		$searchtype = addslashes($searchtype);
		$searchterm = addslashes($searchterm);
	}
	
	switch ($searchtype){
		case "clinic":
			$searchtype = "c.clinicname";
			break;
	
		case "treatment":
			$searchtype = "c.treatment";
			break;
	
		case "doctor":
			$searchtype = "d.doctorname";
			break;
	
		default:
			break;
	}
	
	display_search_results($db, $searchtype, $searchterm);
}

$db->close();
do_footer();
?>