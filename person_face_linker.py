from utilities import Person, get_identity


class LinkedFace(object):
    person_ids: dict

    face_id: int or None

    def __init__(self, identity: str = None):
        self.person_ids = {}
        self.face_id = identity

    def check_if_known_face(self, feed_id: int, personId: int) -> bool:
        if feed_id not in self.person_ids.keys():
            return False
        return personId == self.person_ids[feed_id]

    def register_person(self, person: Person):
        print(f"Registering person {person.track_id}")
        if person.feed_id in self.person_ids.keys():
            print(
                f"Person {person.track_id} already registered as {self.person_ids[person.feed_id]}"
            )
            if self.person_ids[person.feed_id] == person.track_id:
                print("they equal")
                return True  # track ID known, FaceID has been registered already

        print("New DF Scan")
        indentity = get_identity(person)
        if indentity == "":
            return False
        print(f"ident: {indentity}")
        if self.face_id is None:
            self.face_id = indentity  # if no faceID is known set it now

        print(f"Identity is: {indentity}")
        if indentity != self.face_id:
            return False  # Face doesn't match registered identity
        self.person_ids[person.feed_id] = (
            person.track_id
        )  # faceID has been set, register track id for feed
        return True
